from subprocess import check_output

# known device constants
VGA_DEVICE_CLASS = '0300'
NVIDIA_VENDOR_ID = '10de'

class pci_device(object):
    """Represents a device on the pci bus"""

    bus_number = ''
    device_number = ''
    function_number = ''
    device_class = ''
    vendor_id = ''
    device_id = ''

    def __init__(self, line):
        """constructor"""
        self.bus_number = line.split(':', 1)[0]
        self.device_number = line.split(' ')[0].split(':')[1].split('.')[0]
        self.function_number = line.split(' ')[0].split(':')[1].split('.')[1]
        self.device_class = line.split(' ')[1].strip(':')
        self.vendor_id = line.split(' ')[2].split(':')[0]
        self.device_id = line.split(' ')[2].split(':')[1]

    @property
    def slot(self):
        """Constructs a slot from the pci device info"""
        return self.bus_number + ":" + self.device_number + '.' + self.function_number

    @classmethod
    def get_devices(cls):
        std_out = check_output(['lspci', '-n'])
        devices = list()
        for line in std_out.split('\n'):
            if line:
                # create a device from this line, this probably could be more effiecient
                devices.append(pci_device(line))
        return devices


class nvidia_pci_device(pci_device):
    """This holds the meta data for any nvidia device"""

    @classmethod
    def get_nvidia_devices(cls):
        devices = pci_device.get_devices()
        nvidia_devices = list()
        for device in devices:
            if device.vendor_id == NVIDIA_VENDOR_ID:
                nvidia_devices.append(device)
        return nvidia_devices

class nvidia_gpu_pci_device(nvidia_pci_device):
    """This will represent an nvidia GPU pci device"""

    @classmethod
    def get_nvidia_gpu_devices(cls):
        nvidia_devices = nvidia_pci_device.get_nvidia_devices()
        nvidia_gpu_devices = list()
        for nvidia_device in nvidia_devices:
            if nvidia_device.device_class == VGA_DEVICE_CLASS:
                nvidia_gpu_devices.append(nvidia_device)
        return nvidia_gpu_devices

    def get_minimum_bandwidth(self):
        """This will return the minimum bandwith of the GPU's"""

        stdout = check_output('sudo lspci -vvv -s ' + self.slot + ' | grep \'LnkSta.*Width x16\'')

        if not stdout:
            raise Exception("Could not get the minimum bandwidth.")

        return stdout.split(":")[1].split(',')[0]


    def is_alive(self):
        """This will return a safe boolean which tells us if the GPU is still alive."""

        try:
            self.get_minimum_bandwidth()
            #TODO: check the output from the above method to make sure it makes logical sense
            return True
        except Exception:
            return False