#!/usr/bin/env python
# Copyright (C) 2017 Thermo Fisher Scientific. All Rights Reserved.

import dbus, gobject, avahi
from dbus.mainloop.glib import DBusGMainLoop
import threading
import time
import copy
import socket


class IonMeshDiscoveryManager(threading.Thread):
    """This class will monitor avahi via dbus in order to detect the current mesh network setup"""

    # singleton instance
    __singleton = None

    # list of computers in the mesh
    __meshComputers = list()

    # registration type
    __REG_TYPE = '_ionMesh._tcp'

    # thread lock
    __threadLock = threading.Lock()

    # local host name
    __localhost = ''

    def __new__(cls, *args, **kwargs):
        if not cls.__singleton:
            cls.__singleton = super(IonMeshDiscoveryManager, cls).__new__(cls, *args, **kwargs)
            gobject.threads_init()

            # setup dbus stuff
            cls.__singleton.__bus = dbus.SystemBus(mainloop=DBusGMainLoop())
            cls.__singleton.__server = dbus.Interface(cls.__singleton.__bus.get_object(avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER), avahi.DBUS_INTERFACE_SERVER)

            # Look for self.regtype services and hook into callbacks
            cls.__singleton.__browser = dbus.Interface(cls.__singleton.__bus.get_object(avahi.DBUS_NAME, cls.__singleton.__server.ServiceBrowserNew(
                avahi.IF_UNSPEC, avahi.PROTO_UNSPEC, cls.__singleton.__REG_TYPE, "", dbus.UInt32(0))), avahi.DBUS_INTERFACE_SERVICE_BROWSER)
            cls.__singleton.__browser.connect_to_signal("ItemNew", cls.__singleton.__serviceFound)
            cls.__singleton.__browser.connect_to_signal("ItemRemove", cls.__singleton.__serviceRemoved)
            cls.__singleton.__loop = gobject.MainLoop()

            # initialize threading super object
            threading.Thread.__init__(cls.__singleton)
            cls.__singleton.start()

        return cls.__singleton


    def stop(self):
        """
        Call this to stop the thread as one would expect
        """
        self.__loop.quit()


    def run(self):
        """
        Method called with the thread object's "start" method is called
        """
        self.__loop.run()


    def getMeshComputers(self):
        """
        This will create a copy of the list of the comupters in the mesh by hostname
        :return: List of hostnames
        """
        self.__threadLock.acquire()
        try:
            return copy.deepcopy(self.__meshComputers)
        finally:
            self.__threadLock.release()


    def getLocalComputer(self):
        """Gets the localhost name"""
        return self.__localhost or socket.getfqdn()


    def __serviceFound(self, interface, protocol, name, stype, domain, flags):
        """Callback for when a service needs to be added to the mesh list"""
        # skip local services
        # http://sources.debian.net/src/avahi/0.6.32-1/avahi-python/avahi/__init__.py/?hl=50#L50
        if flags & avahi.LOOKUP_RESULT_LOCAL:
            self.__localhost = str(name)
            return

        # add the computer name to the list of mesh computers
        self.__threadLock.acquire()
        try:
            if name not in self.__meshComputers:
                self.__meshComputers.append(str(name))
        finally:
            self.__threadLock.release()


    def __serviceRemoved(self, interface, protocol, name, stype, domain, flags):
        """
        Callback for when a service needs to be removed from the mesh list
        :param interface:
        :param protocol:
        :param name:
        :param stype:
        :param domain:
        :param flags:
        """
        self.__threadLock.acquire()
        try:
            if name in self.__meshComputers:
                self.__meshComputers.remove(str(name))
        finally:
            self.__threadLock.release()


if __name__ == '__main__':
    mesh1 = IonMeshDiscoveryManager()

    try:
        while True:
            time.sleep(1)
            print "*************************"
            print mesh1.getMeshComputers()
    except KeyboardInterrupt:
        pass
    finally:
        mesh1.stop()
