/* A weak definition of the name `common_obj.c` leaves tentative. A strong
 * definition would outrank both; a weak one does not outrank the tentative. */
__attribute__((weak)) int shared_obj = 7;
