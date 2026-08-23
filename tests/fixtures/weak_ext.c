/* Weak definition of the same symbol. A strong definition, if present,
 * must take precedence over this one during resolution. */
__attribute__((weak)) int external_func(void) {
    return 1;
}
