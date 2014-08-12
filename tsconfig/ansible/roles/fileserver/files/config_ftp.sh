#!/bin/bash
function deny_ssh_access ()
{
    deny_user=$1

    if ! grep -q ^DenyUsers /etc/ssh/sshd_config; then
        # DenyUsers directive does not exist; add it
        echo "DenyUsers $deny_user" >> /etc/ssh/sshd_config
    else
        # DenyUsers directive exists; make sure $deny_user is in the list
        if ! echo $directive|grep -q $deny_user; then
            # add $deny_user to list of denied users
            sed -i "/^DenyUsers/ {/$deny_user/! s/.*/& $deny_user/}" /etc/ssh/sshd_config
        fi
    fi
}

#---                                                                                    ---#
#---    Manage the symbolic link from ftp client user home to the raw data directory    ---#
#---                                                                                    ---#
function ionguest_ftp_symlink ()
{
    RAW_DATA_DIR=$1

    if [ -d /home/ionguest ]; then
        if [ -h /home/ionguest/results ]; then
            rm -f /home/ionguest/results
        else
            echo -e "\nSymbolic link does not exist"
        fi
        ln -s ${RAW_DATA_DIR} /home/ionguest/results
        echo -e "\nCreated symlink:\n$(ls -l /home/ionguest/results)\n"
    else
        echo -e "\nUsual ftp client user does not exist: /home/ionguest\n"
    fi
}

function config_ftp()
{

    ftpUser="ionguest"
    ftpPass="ionguest"
    # create ftp user, if it doesn't yet exist
    if ! getent passwd | grep -q "^$ftpUser:"; then
    sudo useradd -m $ftpUser -s /bin/sh
    sudo passwd $ftpUser <<EOFftp
$ftpPass
$ftpPass
EOFftp
    fi

    #Deny ssh access for this user
    deny_ssh_access $ftpUser

#        # disable login shell access
#        # N.B. DO NOT ENABLE!  This diables ftp access
#        status=$(expect -c "
#        spawn chsh -s /bin/false $ftpUser
#        expect {
#        assword: {send \"$ftpPass\n\"; exp_continue}
#        }
#        exit
#        ")

    # this directory is created elsewhere (?) but want to make sure it exists
    # if it exists, make sure it is writeable
    # N.B. cluster setups probably don't write raw data to /results
    #mkdir -p /results
    if [ -d /results ]; then chmod 777 /results || true; fi
    if [ -d /rawdata ]; then chmod 777 /rawdata || true; fi

    # create link to /results; directory where the PGMs write their data files.
    # To support newer hardware, where there is a /rawdata partition we use that
    # partition for raw data symlink, else we default to original /results.
    # /home/ionguest/results -> [ /results | /rawdata ]
    #
    if [ -d /rawdata ]; then
        ionguest_ftp_symlink /rawdata
    else
        ionguest_ftp_symlink /results
    fi

    sed -i "s/^#local_enable.*/local_enable=YES/" /etc/vsftpd.conf
    sed -i "s/^#write_enable.*/write_enable=YES/" /etc/vsftpd.conf
    sed -i "s/^#local_umask.*/local_umask=000/" /etc/vsftpd.conf

    #--- Restrict ftp access to ftp user only  ---#
    # create control file if not existing
    if [ ! -f /etc/vsftpd.allowed_users ]; then
        touch /etc/vsftpd.allowed_users
    fi

    # add ftp user to control file if not existing
    if ! grep -q $ftpUser /etc/vsftpd.allowed_users; then
        echo $ftpUser >> /etc/vsftpd.allowed_users
    fi

    # add control configuration to conf file if not existing
    if ! grep -q ^userlist_deny /etc/vsftpd.conf; then
        echo "userlist_deny=NO" >> /etc/vsftpd.conf
    else
        sed -i "s/userlist_deny.*/userlist_deny=NO/" /etc/vsftpd.conf
    fi
    if ! grep -q ^userlist_enable /etc/vsftpd.conf; then
        echo "userlist_enable=YES" >> /etc/vsftpd.conf
    else
        sed -i "s/userlist_enable.*/userlist_enable=YES/" /etc/vsftpd.conf
    fi
    if ! grep -q ^userlist_file /etc/vsftpd.conf; then
        echo "userlist_file=/etc/vsftpd.allowed_users" >> /etc/vsftpd.conf
    else
        sed -i "s:userlist_file.*:userlist_file=/etc/vsftpd.allowed_users:" /etc/vsftpd.conf
    fi

    service vsftpd restart 
    return 0
}

config_ftp

exit
