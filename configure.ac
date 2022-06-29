AC_INIT([NN-CPU library], [0.1], [arnaucinca1998@gmail.com])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])

AC_LANG(C)

# Prepare compilation flags before AC_PROG_CC
AX_COMPILE_FLAGS

# Look for a valid C compiler
AC_PROG_CC

# AM_PROG_AR must be called before LT_INIT or a warning ensues
AM_PROG_AR

# Automake initialization
AM_INIT_AUTOMAKE([foreign -Wall subdir-objects silent-rules])
AM_SILENT_RULES([yes])

# Checks for header files.
AC_CHECK_HEADERS([unistd.h])
AC_CHECK_HEADERS([xmmintrin.h])

# Checks for typedefs, structures, and compiler characteristics
AC_TYPE_SIZE_T

AC_CHECK_EXTRACT_FIRST_COMPILER_FLAG([FALIGNED_NEW_FLAG], [-faligned-new])

LT_INIT([shared static pic-only])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
