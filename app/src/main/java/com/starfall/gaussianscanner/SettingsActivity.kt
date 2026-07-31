package com.starfall.gaussianscanner

import android.content.SharedPreferences
import android.os.Bundle; import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatActivity
import com.starfall.gaussianscanner.databinding.ActivitySettingsBinding

class SettingsActivity : AppCompatActivity() {
    private lateinit var b: ActivitySettingsBinding
    private lateinit var prefs: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivitySettingsBinding.inflate(layoutInflater); setContentView(b.root)
        prefs = getSharedPreferences("gaussianscanner", MODE_PRIVATE)

        b.rgQuality.setOnCheckedChangeListener { _, id ->
            val v = when (id) {
                R.id.rbLow -> "low"
                R.id.rbHigh -> "high"
                else -> "medium"
            }
            prefs.edit().putString("quality", v).apply()
        }

        val current = prefs.getString("quality", "medium") ?: "medium"
        when (current) {
            "low" -> b.rbLow.isChecked = true
            "high" -> b.rbHigh.isChecked = true
            else -> b.rbMedium.isChecked = true
        }
    }

    companion object {
        fun getTargetWidth(prefs: SharedPreferences): Int = when (prefs.getString("quality", "medium")) {
            "low" -> 300
            "high" -> 800
            else -> 500
        }
    }
}
