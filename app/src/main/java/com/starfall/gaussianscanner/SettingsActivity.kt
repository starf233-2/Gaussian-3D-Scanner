package com.starfall.gaussianscanner

import android.content.SharedPreferences
import android.os.Bundle
import android.view.View
import android.widget.SeekBar
import androidx.appcompat.app.AppCompatActivity
import com.starfall.gaussianscanner.databinding.ActivitySettingsBinding
import com.starfall.gaussianscanner.qnn.QnnEngine
import kotlin.math.sqrt

class SettingsActivity : AppCompatActivity() {
    private lateinit var b: ActivitySettingsBinding
    private lateinit var prefs: SharedPreferences
    private var syncing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivitySettingsBinding.inflate(layoutInflater); setContentView(b.root)
        prefs = getSharedPreferences("gaussianscanner", MODE_PRIVATE)

        // ---- device info ------------------------------------------------
        val di = QnnEngine.detectDevice()
        b.tvDeviceInfo.text = buildString {
            append(di.acceleratorSummary)
            if (di.socModel.isNotBlank()) {
                append(" · ").append(di.socModel)
            }
        }
        b.tvDeviceInfo.text = "${b.tvDeviceInfo.text}\nHTP硬件: ${if (di.hasHtp) "有（精度/驱动需运行时检查）" else "无"} · DSP v66: ${if (di.hasDspV66) "有 (865/870)" else "无"}"
        // server note about NPU skel requirement
        if (di.isQualcomm && !di.hasHtp && !di.hasDspV66) {
            b.tvDeviceInfo.text = "${b.tvDeviceInfo.text}\n提示：该设备暂无可用NPU后端，建议使用GPU加速"
        }

        // ---- engine preference ------------------------------------------
        when (prefs.getString("engine", "auto")) {
            "gpu" -> b.rbEngineGpu.isChecked = true
            "npu" -> b.rbEngineNpu.isChecked = true
            "cpu" -> b.rbEngineCpu.isChecked = true
            else -> b.rbEngineAuto.isChecked = true
        }
        b.rgEngine.setOnCheckedChangeListener { _, id ->
            when (id) {
                R.id.rbEngineGpu -> prefs.edit().putString("engine", "gpu").apply()
                R.id.rbEngineNpu -> prefs.edit().putString("engine", "npu").apply()
                R.id.rbEngineCpu -> prefs.edit().putString("engine", "cpu").apply()
                else -> prefs.edit().putString("engine", "auto").apply()
            }
        }

        // ---- GPU precision ----------------------------------------------
        when (prefs.getString("gpu_quality", "balanced")) {
            "fast" -> b.rbGpuFast.isChecked = true
            "precise" -> b.rbGpuPrecise.isChecked = true
            else -> b.rbGpuBalanced.isChecked = true
        }
        b.rgGpuQuality.setOnCheckedChangeListener { _, id ->
            when (id) {
                R.id.rbGpuFast -> prefs.edit().putString("gpu_quality", "fast").apply()
                R.id.rbGpuPrecise -> prefs.edit().putString("gpu_quality", "precise").apply()
                else -> prefs.edit().putString("gpu_quality", "balanced").apply()
            }
        }

        // ---- GPU delegate range -----------------------------------------
        when (prefs.getString("gpu_mode", "auto")) {
            "quality" -> b.rbGpuModeQuality.isChecked = true
            else -> b.rbGpuModeAuto.isChecked = true
        }
        b.rgGpuMode.setOnCheckedChangeListener { _, id ->
            when (id) {
                R.id.rbGpuModeQuality -> prefs.edit().putString("gpu_mode", "quality").apply()
                else -> prefs.edit().putString("gpu_mode", "auto").apply()
            }
        }

        // ---- GPU 兼容模式 ------------------------------------------------
        b.swGpuCompat.isChecked = prefs.getBoolean("gpu_compat", false)
        b.swGpuCompat.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("gpu_compat", checked).apply()
        }

        // ---- 观感/精度后处理 ---------------------------------------------
        b.swDepthSmooth.isChecked = prefs.getBoolean("depth_smooth", true)
        b.swDepthSmooth.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("depth_smooth", checked).apply()
        }
        b.swColorEnhance.isChecked = prefs.getBoolean("color_enhance", true)
        b.swColorEnhance.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("color_enhance", checked).apply()
        }
        b.swLighting.isChecked = prefs.getBoolean("lighting", true)
        b.swLighting.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("lighting", checked).apply()
        }

        // ---- 手动输入焦距 ------------------------------------------------
        b.swManualFocal.isChecked = prefs.getBoolean("manual_focal", false)
        b.swManualFocal.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("manual_focal", checked).apply()
        }

        // ---- 自动点大小 --------------------------------------------------
        b.swAutoPtSize.isChecked = prefs.getBoolean("auto_pt_size", false)
        b.swAutoPtSize.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("auto_pt_size", checked).apply()
        }

        // ---- 高分辨率推理（切块融合） ------------------------------------
        b.swTiled.isChecked = prefs.getBoolean("tiled_inference", false)
        b.swTiled.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("tiled_inference", checked).apply()
        }

        val current = prefs.getString("quality", "medium") ?: "medium"
        when (current) {
            "low" -> b.rbLow.isChecked = true
            "high" -> b.rbHigh.isChecked = true
            "custom" -> b.rbCustom.isChecked = true
            else -> b.rbMedium.isChecked = true
        }
        updateCustomPanel()

        val savedPoints = prefs.getInt("custom_points", 500).coerceIn(40, 1200)
        val savedWidth = prefs.getInt("custom_width", 612).coerceIn(100, 1000)
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
            updateDebugPanel()
        }
        updateDebugPanel()

        b.swDebugLog.isChecked = prefs.getBoolean("debug_log", false)
        b.swDebugLog.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("debug_log", checked).apply()
        }

        b.swDepthRay.isChecked = prefs.getBoolean("depth_ray", false)
        b.swDepthRay.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("depth_ray", checked).apply()
        }

        b.swShowPtCount.isChecked = prefs.getBoolean("show_pt_count", false)
        b.swShowPtCount.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("show_pt_count", checked).apply()
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

        val savedRadius = getFuzzyRadius(prefs)
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

    private fun updateDebugPanel() {
        b.debugPanel.visibility = if (prefs.getBoolean("debug_mode", false))
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
        b.tvPointsValue.text = if (k >= 1000) "${k / 1000}.${(k % 1000) / 100}M" else "${k}k"
        b.tvWidthValue.text = "${100 + b.sbWidth.progress}"
    }

    companion object {
        fun getEngine(prefs: SharedPreferences): String =
            prefs.getString("engine", "auto") ?: "auto"

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
                "custom" -> prefs.getInt("custom_points", 500).coerceIn(40, 1200) * 1000
                else -> 250000
            }
        }

        fun computeCustomWidth(prefs: SharedPreferences, bw: Int, bh: Int): Int {
            val points = getMaxPoints(prefs)
            val aspect = bw.toDouble() / maxOf(bh.toDouble(), 1.0)
            // No 100px minimum: a very tall image would otherwise allocate
            // millions of unused pixels before the hard point cap was applied.
            val maxWidth = minOf(2000, (4096 * aspect).toInt().coerceAtLeast(1))
            return sqrt(points * aspect).toInt().coerceIn(1, maxWidth)
        }

        fun getFuzzyRadius(prefs: SharedPreferences): Float =
            prefs.getFloat("fuzzy_radius", 1.0f).let { if (it.isFinite()) it.coerceIn(0.4f, 3.0f) else 1f }

        fun getEdgeTrim(prefs: SharedPreferences): Pair<Boolean, Float> {
            val enabled = prefs.getBoolean("edge_trim", false)
            val strength = prefs.getInt("edge_trim_strength", 10).coerceIn(0, 20) / 20f
            return Pair(enabled, strength)
        }

        fun getDebugMode(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("debug_log", false)

        fun getDebugPanelOn(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("debug_mode", false)

        fun getUseFocal(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("use_focal", true)

        fun getManualFocal(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("manual_focal", false)

        /** Last focal length entered by the user, in millimeters. Default 45mm. */
        fun getManualFocalMm(prefs: SharedPreferences): Float =
            prefs.getFloat("manual_focal_mm", 45f).let { if (it.isFinite()) it.coerceIn(1f, 200f) else 45f }

        /**
         * Convert a 35mm-equivalent focal length (mm) to pixels for an image of
         * the given width, using a full-frame sensor width of 36mm.
         */
        fun manualFocalPx(prefs: SharedPreferences, imageWidth: Int): Float {
            if (!getManualFocal(prefs)) return 0f
            val mm = getManualFocalMm(prefs)
            return mm / 36f * imageWidth
        }

        fun getDepthInterp(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("depth_interp", true)

        fun getAutoPtSize(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("auto_pt_size", false)

        fun getDepthRay(prefs: SharedPreferences): Boolean =
            prefs.getBoolean("depth_ray", false)
    }
}
