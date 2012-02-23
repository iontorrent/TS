# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from zeroinstall.injector.config import load_config
from zeroinstall.support import tasks


import sys
import os

from zeroinstall import helpers
from zeroinstall import zerostore
from zeroinstall.injector.model import network_minimal

from zeroinstall.injector import handler
from zeroinstall.injector import trust

trust_db = trust.TrustDB()

config = load_config()
config.network_use=network_minimal
config.feed_mirror = None

# Reads from ~/.config/0install.net/implementation-dirs - default empty
# Just take default and manually append
stores = zerostore.Stores()
pluginStore = zerostore.Store('/results/plugins/implementations')
stores.stores.append(pluginStore)

class AntiTrustMgr(trust.TrustMgr):
    """
    subclass trust.TrustMgr so we can replace the confim_keys method to accept our keys without input
    """

    def confirm_keys(self, pending):

        assert pending.sigs

        from zeroinstall.injector import gpg
        valid_sigs = [s for s in pending.sigs if isinstance(s, gpg.ValidSig)]
        if not valid_sigs:
            def format_sig(sig):
                msg = str(sig)
                if sig.messages:
                    msg += "\nMessages from GPG:\n" + sig.messages
                return msg
            raise SafeException(_('No valid signatures found on "%(url)s". Signatures:%(signatures)s') %
                    {'url': pending.url, 'signatures': ''.join(['\n- ' + format_sig(s) for s in pending.sigs])})

        domain = trust.domain_from_url(pending.url)

        for sig in valid_sigs:
            if self.config.auto_approve_keys:
                existing_feed = self.config.iface_cache.get_feed(pending.url)
                if not existing_feed:
                    trust_db.trust_key(sig.fingerprint, domain)
                    trust_db.notify()

        # Take the lock and confirm this feed
        self._current_confirm = lock = tasks.Blocker('confirm key lock')
        self._current_confirm = None
        lock.trigger()

class NoVerifyHandler(handler.ConsoleHandler):

    def confirm_import_feed(self, pending, valid_sigs):
        """
        verify the feed
        """
        from zeroinstall.injector import trust

        assert valid_sigs

        domain = trust.domain_from_url(pending.url)

        # Ask on stderr, because we may be writing XML to stdout
        print "Feed: %s" % pending.url
        print "The feed is correctly signed with the following keys:"
        for x in valid_sigs:
            print "-", x

        def text(parent):
            text = ""
            for node in parent.childNodes:
                if node.nodeType == node.TEXT_NODE:
                    text = text + node.data
            return text

        shown = set()
        key_info_fetchers = valid_sigs.values()
        while key_info_fetchers:
            old_kfs = key_info_fetchers
            key_info_fetchers = []
            for kf in old_kfs:
                infos = set(kf.info) - shown
                if infos:
                    if len(valid_sigs) > 1:
                        print("%s: " % kf.fingerprint)
                    for key_info in infos:
                        print("-", text(key_info) )
                        shown.add(key_info)
                if kf.blocker:
                    key_info_fetchers.append(kf)
            if key_info_fetchers:
                for kf in key_info_fetchers: print(kf.status)
                stdin = tasks.InputBlocker(0, 'console')
                blockers = [kf.blocker for kf in key_info_fetchers] + [stdin]
                yield blockers
                for b in blockers:
                    try:
                        tasks.check(b)
                    except Exception as ex:
                        warn(_("Failed to get key info: %s"), ex)
                if stdin.happened:
                    print("Skipping remaining key lookups due to input from user")
                    break

        for key in valid_sigs:
            print("Trusting %(key_fingerprint)s for %(domain)s") % {'key_fingerprint': key.fingerprint, 'domain': domain}
            trust.trust_db.trust_key(key.fingerprint, domain)

handler.Handler = NoVerifyHandler
trust.TrustMgr = AntiTrustMgr

def ZeroFindPath(url):
    """
    find the path on the file system from the url
    from https://github.com/gfxmonk/0find/blob/master/zerofind.py
    """
    try:
        del os.environ['DISPLAY']
    except KeyError: pass
    selections = helpers.ensure_cached(url, command=None)
    if not selections:
        return None
    selection = selections.selections[url]

    if selection.id.startswith("package:"):
        print >> sys.stderr, "Package implementation: %s" % (selection.id,)
        return None
    if os.path.exists(selection.id):
        return selection.id
    return zerostore.Stores().lookup_any(selection.digests)


@tasks.async
def download_info(feed_url):
    print "Downloading", feed_url

    feed_download = config.fetcher.download_and_import_feed(feed_url)
    yield feed_download
    tasks.check(feed_download)

    #print "Download complete"

    feed = config.iface_cache.get_feed(feed_url)
    #print "Name:", feed.name
    #print "Summary:", feed.summary

def getFeedName(url):
    feed = config.iface_cache.get_feed(url)
    return str(feed.name)

def getFeed(url):
    return config.iface_cache.get_feed(url)

def downloadZeroFeed(url):
    tasks.wait_for_blocker(download_info(url))

    try:
        zeroPath = ZeroFindPath(url)
        return zeroPath
    except:
        return False


if __name__ == '__main__':
    """for command line testing"""
    print "installing" + sys.argv[1]
    print "WAITING"
    path = downloadZeroFeed(sys.argv[1])
    print "DONE"
    print path
