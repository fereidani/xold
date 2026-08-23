// Process entry point for the static-link smoke test.
//
// Calls `entry`, then invokes the Linux `exit` syscall with the return value
// so the linked program's exit code equals `entry()`'s result.

    .text
    .globl _start
_start:
    call    entry
    movl    %eax, %edi
    movl    $60, %eax
    syscall
