/* Freestanding program for the static-link smoke test.
 *
 * Compiled non-PIE so global access is PC-relative (or absolute), not routed
 * through the GOT. `_start` (in start.S) calls `entry` and exits with its
 * return value, so a successful link and run leaves the process exit code at
 * `counter + 1`. */
volatile int counter = 5;

int entry(void) {
    counter += 1;
    return counter;
}
