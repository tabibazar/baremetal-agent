// disktest.c — does a BareMetal Cloud instance have a writable, persistent disk?
//
// The local BareMetal-App setup attaches a real 512 MB disk.img. The cloud's
// Firecracker config points rootfs at a file called "stub-rootfs.bin", which
// suggests it may not be writable, or may not persist across a restart, or may
// be shared. Guessing is cheap and wrong; this measures it.
//
// Reads a counter from disk, increments it, writes it back, prints both. Run it,
// restart the instance, run it again:
//
//   count = 1 both times  -> writes are lost (or the file never persists)
//   count = 1 then 2      -> the disk is real and persistent
//   open() fails          -> no writable filesystem at all
//
//   ./1-build.sh disktest.c && ./2-run.sh

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#define PATH "/agent_boot_count"

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("DISKTEST_START path=%s\n", PATH);

    long count = 0;

    int fd = open(PATH, O_RDONLY);
    if (fd < 0) {
        printf("read: no existing file (%s)\n", strerror(errno));
    } else {
        char buf[64] = {0};
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n > 0) {
            count = strtol(buf, NULL, 10);
            printf("read: found count=%ld\n", count);
        } else {
            printf("read: file present but empty (n=%zd)\n", n);
        }
    }

    count++;

    fd = open(PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        printf("DISKTEST_RESULT write=FAILED (%s) count_would_be=%ld\n",
               strerror(errno), count);
        printf("verdict: no writable filesystem on this instance\n");
        return 1;
    }

    char out[64];
    int len = snprintf(out, sizeof(out), "%ld\n", count);
    ssize_t w = write(fd, out, (size_t)len);
    int cerr = close(fd);

    if (w != len) {
        printf("DISKTEST_RESULT write=SHORT wrote=%zd of %d close=%d\n", w, len, cerr);
        printf("verdict: filesystem present but writes are not landing\n");
        return 1;
    }

    printf("DISKTEST_RESULT write=OK count=%ld\n", count);
    printf("verdict: wrote successfully. Restart the instance and run again — if the "
           "count increments, the disk persists.\n");
    return 0;
}
