# Ansible Playbook for R&D Servers


## `torrentpy_database.yml`

This playbook is used to modify the default Torrent Server **postgresql** configuration.

1. `/etc/postgresql/9.3/main/postgresql.conf`: set the **listen_addresses** to all (*).

         listen_addresses = '*'

2. `/etc/postgresql/9.3/main/pg_hba.conf`: allow access within internal network.

        host    iondb       ion         10.0.0.0/8            trust
        host    all         all         10.0.0.0/8            trust
        
Prior to restart **postgresql**, all the ion backend daemons need to be shutdown and started afterward.
After setting **listen_addresses** and allowing internal network access, one will need to open up the firewall rules as well. See `open_iptables.yml`.

### Usage

1. copy the playbook (`torrentpy_database.yml`) to the Torrent Server
2. make sure there are no analysis or plugin jobs running.
3. run the playbook with `sudo`:

		sudo ansible-playbook torrentpy_database.yml --sudo \
			-i /usr/share/ion-tsconfig/ansible/torrentsuite_hosts

## `open_iptables.yml`

This playbook is used to open the firewall completely. This should only be used on the server behind the firewall. This is needed for the servers to interact with networked Chef (as opposed to the directly attached ones).

This playbook will simply overwrite `iptables.custom` with the content of `open_iptables.j2` and reload iptable rules without flushing.

### Usage

1. copy the playbook (`open_iptables.yml`) and template (`open_iptables.j2`) to the Torrent Server.
2. Both yml and j2 files should be in the same directory.
3. run the playbook with `sudo`:

		sudo ansible-playbook open_iptables.yml --sudo \
			-i /usr/share/ion-tsconfig/ansible/torrentsuite_hosts

## `sshd_keyonly.yml`

This playbook will add a list of authorized keys to `ionadmin` account from a template (`sshd_keyonly.j2`), which is not committed by choice.

Prior to running this playbook, create a new file with list of authorized public keys. This file will be used to override the exist `~/.ssh/authroized_keys`.

### Usage

1. copy the playbook (`sshd_keyonly.yml`) and template (`sshd_keyonly.j2`) to the Torrent Server.
2. Both yml and j2 files should be in the same directory.
3. run the playbook with `sudo`:

		sudo ansible-playbook sshd_keyonly.yml --sudo \
			-i /usr/share/ion-tsconfig/ansible/torrentsuite_hosts


## `sentry.yml`

This playbook will set up Torrent Browser to be monitored by internal [sentry](https://docs.sentry.io/) server -- sentry.itw.

### Usage

1. copy the playbook (`sentry.yml`) to the Torrent Server
2. make sure there are no analysis or plugin jobs running.
3. run the playbook with `sudo`:

		sudo ansible-playbook sentry.yml --sudo \
			-i /usr/share/ion-tsconfig/ansible/torrentsuite_hosts
