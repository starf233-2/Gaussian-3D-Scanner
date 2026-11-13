package com.starfall.gaussianscanner

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import android.widget.Toast
import java.io.File

/**
 * Saves outputs (depth/ray maps, logs) to PUBLIC directories.
 * Uses MANAGE_EXTERNAL_STORAGE on Android 11+, legacy WRITE on 10-.
 */
object StorageUtil {

    /** whether we can write to public external storage without extra setup. */
    fun canWritePublic(context: Context): Boolean {
        return if (Build.VERSION.SDK_INT >= 30) {
            Environment.isExternalStorageManager()
        } else {
            android.content.pm.PackageManager.PERMISSION_GRANTED ==
                context.checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
    }

    /** request the "All files access" settings screen on Android 11+. */
    fun requestAllFilesAccess(context: Context) {
        if (Build.VERSION.SDK_INT >= 30) {
            try {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                intent.data = Uri.parse("package:${context.packageName}")
                context.startActivity(intent)
            } catch (_: Exception) {
                try {
                    context.startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
                } catch (_: Exception) {
                    Toast.makeText(context, "请授予文件访问权限", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    /** returns the public Pictures/GaussianScanner dir if writable, else app dir. */
    fun picturesDir(context: Context): File {
        val base = if (canWritePublic(context)) {
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        } else {
            context.getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: context.filesDir
        }
        val dir = File(base, "GaussianScanner")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /** returns the public Documents/GaussianScanner dir if writable, else app dir. */
    fun documentsDir(context: Context): File {
        val base = if (canWritePublic(context)) {
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        } else {
            context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
        }
        val dir = File(base, "GaussianScanner")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }
}
