from mininet.topo import Topo

class IoTTopo(Topo):
    def build(self):
        s1 = self.addSwitch('s1')

        h1 = self.addHost('h1')  # Normal IoT device
        h2 = self.addHost('h2')  # Server
        h3 = self.addHost('h3')  # Attacker / faulty device

        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)

topos = {'iottopo': (lambda: IoTTopo())}
