package com.starfall.gaussianscanner.depth

import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean

/** All interpreter/delegate creation, inference and disposal share one native thread.
 * Cancelling a Future does NOT interrupt a driver/JNI call. A timed-out job keeps
 * this lane occupied and disposes its late result here; no parallel fallback is
 * allowed to allocate a second graph or overwrite the active interpreter.
 */
internal class SerialNativeRunner {
    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "DA3-native").apply { isDaemon = true }
    }
    private val occupied = AtomicBoolean(false)
    private val cancellation = ThreadLocal<() -> Boolean>()
    // Process-wide residency as well as execution serialization: a second
    // Activity must not retain a second multi-GB graph alongside the first.
    private var resident: AutoCloseable? = null
    val busy: Boolean get() = occupied.get()

    fun checkActive() {
        if (cancellation.get()?.invoke() == true) throw CancellationException("原生任务已取消")
    }

    fun clearResident() {
        val old = resident
        resident = null
        old?.close()
    }

    fun keepResident(value: AutoCloseable) {
        check(resident == null)
        resident = value
    }

    fun forgetResident(value: AutoCloseable) { if (resident === value) resident = null }

    fun <T> call(timeoutMs: Long, discard: (T) -> Unit = {}, onAbandoned: () -> Unit = {}, block: () -> T): T {
        check(occupied.compareAndSet(false, true)) {
            "上一项原生任务尚未退出，请稍后重试（未启动并行回退）"
        }
        val lock = Object()
        var completed = false
        var abandoned = false
        var result: T? = null
        var failure: Throwable? = null
        try {
            executor.execute {
                var handedOff = false
                cancellation.set { synchronized(lock) { abandoned } }
                try {
                    val value = block()
                    val drop = synchronized(lock) {
                        if (abandoned) true else {
                            result = value
                            occupied.set(false)
                            completed = true
                            handedOff = true
                            lock.notifyAll()
                            false
                        }
                    }
                    if (drop) discard(value)
                } catch (t: Throwable) {
                    synchronized(lock) {
                        failure = t
                        if (!abandoned) {
                            occupied.set(false)
                            completed = true
                            handedOff = true
                            lock.notifyAll()
                        }
                    }
                } finally {
                    try {
                        if (!handedOff) {
                            try { onAbandoned() } finally { occupied.set(false) }
                        }
                    } finally { cancellation.remove() }
                }
            }
        } catch (t: Throwable) {
            occupied.set(false)
            throw t
        }
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs)
        synchronized(lock) {
            try {
                while (!completed) {
                    val remaining = deadline - System.nanoTime()
                    if (remaining <= 0L) {
                        abandoned = true
                        throw TimeoutException("原生任务超时；驱动退出前不再加载其他模型")
                    }
                    TimeUnit.NANOSECONDS.timedWait(lock, remaining)
                }
            } catch (t: InterruptedException) {
                if (completed) Thread.currentThread().interrupt() else {
                    abandoned = true
                    throw t
                }
            }
            failure?.let { throw it }
            @Suppress("UNCHECKED_CAST")
            return result as T
        }
    }

    /** Ordered after any in-flight inference, never closes an interpreter on UI. */
    fun release(block: () -> Unit) {
        executor.execute { runCatching(block) }
    }
}

internal val NativeRunner = SerialNativeRunner()
