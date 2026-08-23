/* A tentative definition (a "common" symbol). ELF precedence gives it the
 * storage over a weak definition of the same name, which is the case
 * `common_outranks_weak_definition` pins down. */
int shared_obj;
