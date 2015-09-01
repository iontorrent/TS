// feature test for kill()
#define _POSIX_SOURCE

#include <sys/types.h>
#include <errno.h>
#include <stdio.h>
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h> 

// ---------- tunnel creation ----------

// second child process
// it's parent becomes init as soon as it's original parent (i.e., the first child process) calls exit()
static void secondChild(
		char const * const host,
		char const * const user,
		const int fd,
		const int connectOnPort,
		const int tunnelEndpointPort)
{
	char fileDescString[32] = {};
	snprintf(fileDescString, 32, "-d%d", fd);

	char connectPortString[32] = {};
	snprintf(connectPortString, 32, "%d", connectOnPort);

	char endpointPortString[32] = {};
	snprintf(endpointPortString, 32, "%d:localhost:22", tunnelEndpointPort);

	// Issue the following ssh command to create an ssh tunnel from localhost port 22  to 'host' port 'port'
	// ssh -N -p 22 -l user -o StrictHostKeyChecking=no -R 15022:localhost:22 host
	// Leaving off the -f option as this causes issues with some proxies.
	int rc = execlp("/usr/bin/sshpass", "/usr/bin/sshpass",
			fileDescString,
			"/usr/bin/ssh",
			"-N",
			"-p", connectPortString,
			"-l", user,
			"-o", "StrictHostKeyChecking=no",
			"-R", endpointPortString,
			host,
			NULL);

	if (rc)
		fprintf(stderr, "%s: execlp: %s\n", __FUNCTION__, strerror(errno));

	exit(EXIT_FAILURE); // won't get here in success case
}

// fork twice so that the ssh tunnel process will be the child of init instead of the child of RSMAgent.
static void firstChild(
		char const * const host,
		char const * const user,
		char const * const pass,
		const int connectOnPort,
		const int tunnelEndpointPort)
{
	// Per sshpass suggestion, use an anonymous pipe to allow sshpass to read the password from a file descriptor.
	const int readEnd = 0;
	const int writeEnd = 1;
	int p2c[2];
	int rc = pipe(p2c);
	if (rc == -1)
		fprintf(stderr, "%s: pipe: %s\n", __FUNCTION__, strerror(errno));

	pid_t pid = fork();

	if (pid == 0) {				// second child process
		rc = close(p2c[writeEnd]);
		if (rc == -1)
			fprintf(stderr, "%s: close: %s\n", __FUNCTION__, strerror(errno));

		secondChild(host, user, p2c[readEnd], connectOnPort, tunnelEndpointPort);
	}
	else if (pid > 0) {			// still first child process (parent of this fork)
		rc = close(p2c[readEnd]);
		if (rc == -1)
			fprintf(stderr, "%s: close: %s\n", __FUNCTION__, strerror(errno));

		// write the password to the pipe so sshpass can read it directly from memory
		ssize_t nbytes = write(p2c[writeEnd], pass, strlen(pass));
		if (nbytes == -1)
			fprintf(stderr, "%s: write: %s\n", __FUNCTION__, strerror(errno));

		exit(EXIT_SUCCESS);
	}
	else if (pid < 0) {			// fork error
		fprintf(stderr, "%s: second fork: %s\n", __FUNCTION__, strerror(errno));
	}
}

// usual C return value: 0 for success, non-zero for failure.
int ssh_tunnel_create(
		char const * const host,
		char const * const user,
		char const * const pass,
		const int connectToPort,
		const int tunnelEndpointPort)
{
	if (!host || !user || !pass) {
		return EINVAL; // host or user or pass was invalid, i.e., NULL
	}

	int retval = 0;
	pid_t pid = fork();

	if (pid == 0) {				// first child process
		firstChild(host, user, pass, connectToPort, tunnelEndpointPort);
	}
	else if (pid > 0) {			// parent
		// Parent waits for *first* child to exit, which happens promptly.
		// The *second* child is the long-running process, and init will reap that process on exit.
		pid_t childPid = waitpid(pid, NULL, 0);
		if (childPid != pid)
			fprintf(stderr, "%s: waitpid: %s\n", __FUNCTION__, strerror(errno));
	}
	else if (pid < 0) {			// fork error
		fprintf(stderr, "%s: first fork: %s\n", __FUNCTION__, strerror(errno));
		retval = 1;
	}
	return retval;
}                

// ---------- tunnel removal ----------

// Return first pid encountered that is for an ssh tunnel process.
static pid_t findPidForSshTunnel()
{
	pid_t pidout = -1;
	DIR* dir = opendir("/proc");
	if (!dir) {
		fprintf(stderr, "%s: opendir: %s\n", __FUNCTION__, strerror(errno));
		return -1;
	}

	struct dirent *entry = NULL;
	do {
		entry = readdir(dir);
		if (!entry)
			break;

		// We only want to search the directories with numbers (i.e., process IDs) for names.
		char* endptr;
		pid_t pid = strtol(entry->d_name, &endptr, 10);
		// if endptr is not a null character, the directory name is not entirely numeric, so skip it.
		if (*endptr != '\0')
			continue;

		char buf[512];
		snprintf(buf, sizeof(buf), "/proc/%d/cmdline", pid);
		FILE* fp = fopen(buf, "r");
		if (fp) {
			memset(buf, 0, 512);
			int rc __attribute__((unused)) = fread(buf, sizeof(char), 512, fp);
			char const * const argv[] = {
			"/usr/bin/sshpass",
			"-d",					// will be followed by integer (file descriptor)
			"/usr/bin/ssh",
			"-N",
			"-p", "*",				// * means accept whatever is in this arg (contains connectToPort)
			"-l", "*",				// * means accept whatever is in this arg (contains user name)
			"-o", "StrictHostKeyChecking=no",
			"-R", ":localhost:22"	// omitting variable port number prefix
			};
			const int argc = sizeof(argv) / sizeof(argv[0]);

			char *cp = buf;
			int good = 1;
			for (int ii = 0; good && ii < argc; ++ii) {
				if (argv[ii][0] != '*') {
					good = good && strstr(cp, argv[ii]) != NULL;
					//if (good) printf("cp %s len %ld argv %s good %d\n", cp, strlen(cp), argv[ii], good);
				}
				cp += strlen(cp) + 1;
			}
			if (good) {
				entry = NULL; // exit the do-while loop
				pidout = pid;
			}

			fclose(fp);
		}
	} while (entry);

	closedir(dir);
	return pidout;
}

void ssh_tunnel_remove()
{
	pid_t pid = 0;

	// Remove all tunnel processes.
	// Send a SIGKILL (kill -9) because in testing I saw ssh tunnels that did not die when sent a SIGTERM.
	while (pid != -1) {
		pid = findPidForSshTunnel();
		if (pid != -1) {
			int rc = kill(pid, SIGKILL); // kill -9
			if (rc == -1)
				fprintf(stderr, "%s: kill: %s\n", __FUNCTION__, strerror(errno));
		}
	}
}

