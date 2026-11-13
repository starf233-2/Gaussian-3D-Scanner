package com.starfall.gaussianscanner.depth

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

class NativeRunnerTest {
    @Test fun createRunAndReleaseUseTheSameThread() {
        val runner = SerialNativeRunner()
        val buildThread = runner.call(1000) { Thread.currentThread().id }
        val runThread = runner.call(1000) { Thread.currentThread().id }
        val released = CountDownLatch(1)
        var releaseThread = 0L
        runner.release { releaseThread = Thread.currentThread().id; released.countDown() }
        assertTrue(released.await(2, TimeUnit.SECONDS))
        assertEquals(buildThread, runThread)
        assertEquals(buildThread, releaseThread)
    }

    @Test fun timeoutCannotStartParallelFallbackAndLateResultIsDisposed() {
        val runner = SerialNativeRunner()
        val finishNative = CountDownLatch(1)
        val discarded = CountDownLatch(1)
        val cleanupDone = CountDownLatch(1)
        var taskThread = 0L; var discardThread = 0L
        assertThrows(TimeoutException::class.java) {
            runner.call(30, discard = { _: Int -> discardThread = Thread.currentThread().id; discarded.countDown() }) {
                taskThread = Thread.currentThread().id
                finishNative.await(2, TimeUnit.SECONDS)
                42
            }
        }
        try {
            assertTrue(runner.busy)
            assertThrows(IllegalStateException::class.java) { runner.call(1000) { error("Must not allocate fallback") } }
        } finally { finishNative.countDown() }
        runner.release { cleanupDone.countDown() }
        assertTrue(cleanupDone.await(2, TimeUnit.SECONDS))
        assertTrue(discarded.await(2, TimeUnit.SECONDS))
        assertEquals(taskThread, discardThread)
        assertEquals(99, runner.call(1000) { 99 })
    }

    @Test fun lateFailureStillReleasesAbandonedInput() {
        val runner = SerialNativeRunner()
        val finish = CountDownLatch(1); val cleanup = CountDownLatch(1); val barrier = CountDownLatch(1)
        assertThrows(TimeoutException::class.java) {
            runner.call<Int>(30, onAbandoned = { cleanup.countDown() }) {
                finish.await(2, TimeUnit.SECONDS)
                error("Late driver error")
            }
        }
        finish.countDown()
        runner.release { barrier.countDown() }
        assertTrue(barrier.await(2, TimeUnit.SECONDS))
        assertTrue(cleanup.await(2, TimeUnit.SECONDS))
        assertFalse(runner.busy)
    }

    @Test fun immediateErrorAndRepeatedHandoffsDoNotLeaveLaneBusy() {
        val runner = SerialNativeRunner()
        assertThrows(IllegalArgumentException::class.java) { runner.call<Int>(1000) { throw IllegalArgumentException("test") } }
        repeat(1000) { assertEquals(it, runner.call(1000) { it }) }
    }

    @Test fun interruptedCallerCannotReleaseNativeInputsEarly() {
        val runner = SerialNativeRunner()
        val started = CountDownLatch(1); val finish = CountDownLatch(1)
        val callerDone = CountDownLatch(1); val inputFreed = CountDownLatch(1); val barrier = CountDownLatch(1)
        var failure: Throwable? = null
        val caller = Thread {
            try {
                runner.call(5000, onAbandoned = { inputFreed.countDown() }) {
                    started.countDown(); finish.await(3, TimeUnit.SECONDS); 1
                }
            } catch (t: Throwable) { failure = t } finally { callerDone.countDown() }
        }
        caller.start()
        assertTrue(started.await(2, TimeUnit.SECONDS))
        caller.interrupt()
        assertTrue(callerDone.await(2, TimeUnit.SECONDS))
        assertTrue(failure is InterruptedException)
        assertEquals(1L, inputFreed.count)
        finish.countDown()
        runner.release { barrier.countDown() }
        assertTrue(barrier.await(2, TimeUnit.SECONDS))
        assertTrue(inputFreed.await(2, TimeUnit.SECONDS))
    }

    @Test fun replacingAnActivityCannotRetainTwoGraphs() {
        val runner = SerialNativeRunner()
        var alive = 0
        runner.call(1000) { runner.keepResident(AutoCloseable { alive-- }); alive++ }
        runner.call(1000) {
            runner.clearResident()
            assertEquals(0, alive)
            alive++
            runner.keepResident(AutoCloseable { alive-- })
        }
        runner.call(1000) { runner.clearResident() }
        assertEquals(0, alive)
    }

    @Test fun cancellationStopsFurtherNativeWorkAfterDriverReturns() {
        val runner = SerialNativeRunner()
        val finish = CountDownLatch(1); val barrier = CountDownLatch(1)
        var secondNativeCall = false
        assertThrows(TimeoutException::class.java) {
            runner.call(30) {
                finish.await(2, TimeUnit.SECONDS)
                runner.checkActive()
                secondNativeCall = true
            }
        }
        finish.countDown()
        runner.release { barrier.countDown() }
        assertTrue(barrier.await(2, TimeUnit.SECONDS))
        assertFalse(secondNativeCall)
    }
}
