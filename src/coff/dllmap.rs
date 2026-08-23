//! Well-known Win32/CRT function-to-DLL name mapping.
//!
//! A COFF object produced by clang `-msvc` encodes a `dllimport` reference as
//! an undefined external `__imp_<func>` (the IAT slot address), but it never
//! records which DLL provides `<func>`: that information lives in import
//! libraries (`.lib` / `.dll.a`), which xold does not read. To bridge the gap,
//! a small built-in table maps the common Win32 exports and the C runtime
//! exports Wine ships to their DLL. An `__imp_` reference whose function is
//! not in the table surfaces as a precise error at import planning time, so
//! the caller learns the missing name rather than linking a broken image.
//!
//! Wine resolves these DLLs at load time: `kernel32.dll` for the Win32 API
//! and `msvcrt.dll` for the C runtime (both ship under
//! `/usr/*/wine/x86_64-windows`). Function-name lookup is exact and
//! case-sensitive on our side; Windows export resolution is itself
//! case-insensitive, so the on-disk name need not match a particular case.

/// The Win32 base API DLL.
const KERNEL32: &[u8] = b"kernel32.dll";
/// The Microsoft C runtime DLL (the UCRT-style export set Wine provides).
const MSVCRT: &[u8] = b"msvcrt.dll";

/// The static (DLL, function) table, kept grouped by DLL for readability.
/// Lookup is by exact function-name match; order does not drive the result.
const TABLE: &[(&[u8], &[u8])] = &[
    // --- kernel32.dll: process, thread, console, file, memory ----------
    (b"CloseHandle", KERNEL32),
    (b"CreateFileA", KERNEL32),
    (b"CreateFileW", KERNEL32),
    (b"CreateThread", KERNEL32),
    (b"ExitProcess", KERNEL32),
    (b"ExitThread", KERNEL32),
    (b"FreeLibrary", KERNEL32),
    (b"GetACP", KERNEL32),
    (b"GetCommandLineA", KERNEL32),
    (b"GetCommandLineW", KERNEL32),
    (b"GetConsoleMode", KERNEL32),
    (b"GetConsoleOutputCP", KERNEL32),
    (b"GetCurrentProcess", KERNEL32),
    (b"GetCurrentProcessId", KERNEL32),
    (b"GetCurrentThreadId", KERNEL32),
    (b"GetEnvironmentVariableA", KERNEL32),
    (b"GetEnvironmentVariableW", KERNEL32),
    (b"GetExitCodeProcess", KERNEL32),
    (b"GetFileType", KERNEL32),
    (b"GetLastError", KERNEL32),
    (b"GetModuleHandleA", KERNEL32),
    (b"GetModuleHandleW", KERNEL32),
    (b"GetProcAddress", KERNEL32),
    (b"GetExitCodeThread", KERNEL32),
    (b"GetProcessHeap", KERNEL32),
    (b"GetStdHandle", KERNEL32),
    (b"GetSystemTimeAsFileTime", KERNEL32),
    (b"HeapAlloc", KERNEL32),
    (b"HeapCreate", KERNEL32),
    (b"HeapFree", KERNEL32),
    (b"HeapReAlloc", KERNEL32),
    (b"HeapSize", KERNEL32),
    (b"InitializeCriticalSection", KERNEL32),
    (b"LoadLibraryA", KERNEL32),
    (b"LoadLibraryW", KERNEL32),
    (b"MultiByteToWideChar", KERNEL32),
    (b"QueryPerformanceCounter", KERNEL32),
    (b"ReadFile", KERNEL32),
    (b"SetConsoleCtrlHandler", KERNEL32),
    (b"SetFilePointerEx", KERNEL32),
    (b"SetLastError", KERNEL32),
    (b"SetUnhandledExceptionFilter", KERNEL32),
    (b"Sleep", KERNEL32),
    (b"VirtualAlloc", KERNEL32),
    (b"VirtualFree", KERNEL32),
    (b"VirtualProtect", KERNEL32),
    (b"WaitForSingleObject", KERNEL32),
    (b"WideCharToMultiByte", KERNEL32),
    (b"WriteConsoleA", KERNEL32),
    (b"WriteConsoleW", KERNEL32),
    (b"WriteFile", KERNEL32),
    // --- msvcrt.dll: C runtime -----------------------------------------
    (b"abort", MSVCRT),
    (b"abs", MSVCRT),
    (b"atexit", MSVCRT),
    (b"atoi", MSVCRT),
    (b"calloc", MSVCRT),
    (b"exit", MSVCRT),
    (b"fclose", MSVCRT),
    (b"fflush", MSVCRT),
    (b"fopen", MSVCRT),
    (b"fprintf", MSVCRT),
    (b"fputc", MSVCRT),
    (b"fputs", MSVCRT),
    (b"fread", MSVCRT),
    (b"free", MSVCRT),
    (b"fwrite", MSVCRT),
    (b"getenv", MSVCRT),
    (b"malloc", MSVCRT),
    (b"memcmp", MSVCRT),
    (b"memcpy", MSVCRT),
    (b"memmove", MSVCRT),
    (b"memset", MSVCRT),
    (b"printf", MSVCRT),
    (b"putc", MSVCRT),
    (b"putchar", MSVCRT),
    (b"puts", MSVCRT),
    (b"qsort", MSVCRT),
    (b"rand", MSVCRT),
    (b"realloc", MSVCRT),
    (b"snprintf", MSVCRT),
    (b"sprintf", MSVCRT),
    (b"srand", MSVCRT),
    (b"sscanf", MSVCRT),
    (b"strcat", MSVCRT),
    (b"strchr", MSVCRT),
    (b"strcmp", MSVCRT),
    (b"strcpy", MSVCRT),
    (b"strlen", MSVCRT),
    (b"strncmp", MSVCRT),
    (b"strncpy", MSVCRT),
    (b"strstr", MSVCRT),
    (b"system", MSVCRT),
    (b"time", MSVCRT),
    (b"tolower", MSVCRT),
    (b"toupper", MSVCRT),
    (b"vsnprintf", MSVCRT),
    (b"vsprintf", MSVCRT),
];

/// The DLL that exports `func`, if it is in the well-known table.
pub fn resolve(func: &[u8]) -> Option<&'static [u8]> {
    TABLE
        .iter()
        .find(|(name, _)| *name == func)
        .map(|(_, d)| *d)
}
