package com.starfall.gaussianscanner

import android.content.SharedPreferences
import android.os.Bundle
import android.view.View
import android.widget.RadioGroup
import android.widget.SeekBar
import androidx.appcompat.app.AppCompatActivity
import com.starfall.gaussianscanner.databinding.ActivitySettingsBinding
import kotlin.math.sqrt

class SettingsActivity : AppCompatActivity() {
    private lateinit var b: ActivitySettingsBinding
    private lateinit var prefs: SharedPreferences
    private var syncing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivitySettingsBinding.inflate(layoutInflater); setContentView(b.root)
        prefs = getSharedPreferences("gaussianscanner", MODE_PRIVATE)

        val current = prefs.getString("quality", "medium") ?: "medium"
        when (current) {
            "low" -> b.rbLow.isChecked = true
            "high" -> b.rbHigh.isChecked = true
            "custom" -> b.rbCustom.isChecked = true
            else -> b.rbMedium.isChecked = true
        }
        updateCustomPanel()

        val savedPoints = prefs.getInt("custom_points", 500)
        val savedWidth = prefs.getInt("custom_width", 612)
        b.sbPoints.progress = savedPoints - 40
        b.sbWidth.progress = savedWidth - 100

        b.swFuzzy.isChecked = prefs.getBoolean("fuzzy", false)
        b.swFuzzy.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("fuzzy", checked).apply()
            updateFuzzyPanel()
        }
        updateFuzzyPanel()

        b.swDebug.isChecked = prefs.getBoolean("debug_mode", false)
        b.swDebug.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("debug_mode", checked).apply()
        }

        b.swFocal.isChecked = prefs.getBoolean("use_focal", true)
        b.swFocal.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("use_focal", checked).apply()
        }

        b.swDepthInterp.isChecked = prefs.getBoolean("depth_interp", true)
        b.swDepthInterp.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("depth_interp", checked).apply()
        }

        b.swEdgeTrim.isChecked = prefs.getBoolean("edge_trim", false)
        b.swEdgeTrim.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("edge_trim", checked).apply()
            updateEdgeTrimPanel()
        }
        updateEdgeTrimPanel()

        val savedEdgeTrim = prefs.getInt("edge_trim_strength", 10)
        b.sbEdgeTrim.progress = savedEdgeTrim
        updateEdgeTrimLabel()
        b.sbEdgeTrim.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                updateEdgeTrimLabel()
                if (fromUser) prefs.edit().putInt("edge_trim_strength", p).apply()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        val savedRadius = prefs.getFloat("fuzzy_radius", 1.0f)
        b.sbFuzzyRadius.progress = ((savedRadius - 0.4f) / 0.1f).toInt()
        updateFuzzyRadiusLabel()
        b.sbFuzzyRadius.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                updateFuzzyRadiusLabel()
                if (fromUser) prefs.edit().putFloat("fuzzy_radius", 0.4f + p * 0.1f).apply()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        b.rgQuality.setOnCheckedChangeListener { _, id ->
            when (id) {
                R.id.rbLow -> savePreset("low")
                R.id.rbHigh -> savePreset("high")
                R.id.rbCustom -> { prefs.edit().putString("quality", "custom").apply(); updateCustomPanel() }
                else -> savePreset("medium")
            }
        }

        b.sbPoints.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                if (!fromUser || syncing) return
                val points = 40 + p
                syncing = true
                b.sbWidth.progress = widthFromPoints(points) - 100
                syncing = false
                updateLabels()
                prefs.edit()
                    .putInt("custom_points", points)
                    .putInt("custom_width", widthFromPoints(points))
                    .apply()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        b.sbWidth.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                if (!fromUser || syncing) return
                val width = 100 + p
                syncing = true
                b.sbPoints.progress = pointsFromWidth(width) - 40
                syncing = false
                updateLabels()
                prefs.edit()
                    .putInt("custom_width", width)
                    .putInt("custom_points", pointsFromWidth(width))
                    .apply()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        updateLabels()
    }

    private fun widthFromPoints(pointsK: Int): Int {
        // standard portrait 4:3 → bw/bh = 0.75; width = sqrt(points * 0.75)
        return (sqrt(pointsK * 1000.0 * 0.75)).toInt().coerceIn(100, 1000)
    }

    private fun pointsFromWidth(width: Int): Int {
        // points = width^2 / 0.75
        return (width * width / 0.75 / 1000).toInt().coerceIn(40, 1200)
    }

    private fun savePreset(v: String) {
        prefs.edit().putString("quality", v).apply()
        updateCustomPanel()
    }

    private fun updateCustomPanel() {
        b.customPanel.visibility = if (prefs.getString("quality", "medium") == "custom")
            View.VISIBLE else View.GONE
    }

    private fun updateFuzzyPanel() {
        b.fuzzyRadiusPanel.visibility = if (prefs.getBoolean("fuzzy", false))
            View.VISIBLE else View.GONE
    }

    private fun updateEdgeTrimPanel() {
        b.edgeTrimPanel.visibility = if (prefs.getBoolean("edge_trim", false))
            View.VISIBLE else View.GONE
    }

    private fun updateEdgeTrimLabel() {
        val pct = b.sbEdgeTrim.progress * 5
        b.tvEdgeTrimValue.text = "$pct%"
    }

    private fun updateFuzzyRadiusLabel() {
        val r = 0.4f + b.sbFuzzyRadius.progress * 0.1f
        b.tvFuzzyRadiusValue.text = "%.1f".format(r)
    }

    private fun updateLabels() {
        val k = 40 + b.sbPoints.progress
        b.tvPointsValue.text = if (k >= 1000) "${k / 1000}.${(k % 1000) / 100}k" else "${k}k"
        b.tvWidthValue.text = "${100 + b.sbWidth.progress}"
    }

    companion object {
        fun getTargetWidth(prefs: SharedPreferences): Int {
            return when (prefs.getString("quality", "medium")) {
                "low" -> 0
                "high" -> 0
                "custom" -> -1 // computed at build time from points + actual aspect
                else -> 0
            }
        }

        fun getMaxPoints(prefs: SharedPreferences): Int {
            return when (prefs.getString("quality", "medium")) {
                "low" -> 90000
                "high" -> 640000
                "custom" -> (prefs.getInt("custom_points", 500) * 1000).coerceIn(40000, 1200000)
                else -> 250000
            }
        }

        fun computeCustomWidth(prefs: SharedPreferences, bw: Int, bh: Int): Int {
            val points = getMaxPoints(prefs)
            val aspect = bw.toDouble() / maxOf(bh.toDouble(), 1.0)
            return sqrt(points * aspect).toInt().coerceIn(100, 2000)
        }

        fun getFuzzyRadius(prefs: SharedPreferences): Float =
            prefs.getFloat("fuzzy_radius", 1.0f).coerceIn(0.4f, 3.0f)

        fun getEdgeTrim(prefs: SharedPreferences): Pair<Boolean, Float> {
            val enabled = prefs.getBoolean("edge_trim", false)
            val strength = prefs.getInt("edge_trim_strength", 10).coerceIn(0, 20) / 20f
            return Pair(enabled, strength)
        }

        fun getDebugMode(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("debug_mode", false)

        fun getUseFocal(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("use_focal", true)

        fun getDepthInterp(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("depth_interp", true)
    }
}
