# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
"""
NOTE: The GPG algorithm is sensitive to the encoding so if the strings are encoded in something
different than what the django locale's preferred encoding then it will fail to decrypt
"""

from django.db import models
from django.utils.functional import cached_property
import gnupg

HOME = '/tmp'

# read the super secret key from the file system
with open('/var/spool/ion/key') as key_fp:
    SUPER_SECRET_KEY = key_fp.read().strip()

class SecureString(models.Model):
    """Used for storing information securely"""

    # storage for the encrypted password
    # NEVER NEVER ASSIGN THIS FIELD MANUALLY!!!!!!!!!!!!!!
    encrypted_string = models.CharField(null=False, max_length=1000)

    # the date/time this was created
    created = models.DateTimeField(auto_now_add=True)

    # the name and natural key of the entry
    name = models.CharField(max_length=128, unique=True, null=False)

    @classmethod
    def create(cls, unencrypted, name):
        gpg = gnupg.GPG(gnupghome=HOME)
        ep = gpg.encrypt(data=str(unencrypted), recipients=None, symmetric='AES256', passphrase=SUPER_SECRET_KEY)
        if not ep:
            raise Exception("Failed to encrypt the data.")

        return cls(name=name, encrypted_string=ep.data.strip())

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        """Override the save method to make sure the string is correctly encrypted before saving"""

        if not self.encrypted_string:
            raise Exception("Cannot save an empty string for " + self.name)

        if not self.encrypted_string.startswith('-----BEGIN PGP MESSAGE-----'):
            raise Exception("This is not a pgp encrypted string.")

        super(SecureString, self).save(*args, **kwargs)

    def natural_key(self):
        """Get the natural key for this entry"""
        return self.name

    def encrypt(self, unencrypted):
        """Used to update the encrypted string"""
        gpg = gnupg.GPG(gnupghome=HOME)
        ep = gpg.encrypt(data=str(unencrypted), recipients=None, symmetric='AES256', passphrase=SUPER_SECRET_KEY)
        if not ep:
            raise Exception("Failed to encrypt the data.")
        self.encrypted_string = ep.data.strip()

    @cached_property
    def decrypted(self):
        """This will decrypt a encrypted message and make sure the unencrypted string never hits the disk"""
        return str(gnupg.GPG(gnupghome=HOME).decrypt(self.encrypted_string, passphrase=SUPER_SECRET_KEY))
