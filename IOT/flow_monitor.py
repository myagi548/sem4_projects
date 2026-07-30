from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, ether_types
from ryu.lib import hub
from datetime import datetime
import time

MONITOR_INTERVAL = 5

LOW_THRESHOLD = 20
MEDIUM_THRESHOLD = 50
HIGH_THRESHOLD = 100

BLOCK_IDLE_TIMEOUT = 30   # Remove block if no traffic for 30 sec
BLOCK_HARD_TIMEOUT = 60   # Remove block after 60 sec anyway


class IoTMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IoTMonitor, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.flow_stats = {}
        self.blocked_flows = {}  # Track blocked flows
        self.alert_count = 0
        self.monitor_thread = hub.spawn(self.monitor)
        print("\n🚀 SDN-IoT Anomaly Detection Started...\n")

    # Switch connection
    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[datapath.id] = datapath
            print("🔌 Switch Connected Successfully!\n")
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(datapath.id, None)

    # Table-miss rule
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        mod = parser.OFPFlowMod(datapath=datapath,
                                priority=0,
                                match=match,
                                instructions=inst)
        datapath.send_msg(mod)

    # Learning switch
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if ip_pkt:
            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=ip_pkt.src,
                ipv4_dst=ip_pkt.dst
            )

            inst = [parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS, actions)]

            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=1,
                match=match,
                instructions=inst)

            datapath.send_msg(mod)

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data)
        datapath.send_msg(out)

    # Monitoring loop with automatic unblock
    def monitor(self):
        while True:
            now = time.time()
            # Remove expired blocked flows
            for flow_id, block_time in list(self.blocked_flows.items()):
                if now - block_time > BLOCK_HARD_TIMEOUT:
                    del self.blocked_flows[flow_id]
                    print(f"✅ Flow {flow_id} unblocked after timeout")

            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(MONITOR_INTERVAL)

    def request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # Process flow statistics
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply(self, ev):
        datapath = ev.msg.datapath

        for stat in ev.msg.body:
            if stat.priority != 1:
                continue

            src = stat.match.get('ipv4_src')
            dst = stat.match.get('ipv4_dst')

            if not src or not dst:
                continue

            flow_id = (src, dst)

            # Remove expired blocked flows before checking severity
            now = time.time()
            for f, block_time in list(self.blocked_flows.items()):
                if now - block_time > BLOCK_HARD_TIMEOUT:
                    del self.blocked_flows[f]

            current_packets = stat.packet_count
            prev_packets = self.flow_stats.get(flow_id, 0)
            rate = (current_packets - prev_packets) / MONITOR_INTERVAL
            self.flow_stats[flow_id] = current_packets

            print("--------------------------------------------------")
            print(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"📤 Source: {src}")
            print(f"📥 Destination: {dst}")
            print(f"⚡ Packet Rate: {rate:.2f} packets/sec")

            if flow_id in self.blocked_flows:
                severity = "BLOCKED"

            elif rate > HIGH_THRESHOLD:
                severity = "HIGH"
                self.block_flow(datapath, stat.match)
                self.blocked_flows[flow_id] = time.time()
                self.alert_count += 1

            elif rate > MEDIUM_THRESHOLD:
                severity = "MEDIUM"

            elif rate > LOW_THRESHOLD:
                severity = "LOW"

            else:
                severity = "NORMAL"

            print(f"🚦 Severity Level: {severity}")

            if severity in ["HIGH", "MEDIUM", "LOW"]:
                print(f"🔔 Total Alerts: {self.alert_count}")

            print("--------------------------------------------------\n")

    # Temporary block rule
    def block_flow(self, datapath, match):
        parser = datapath.ofproto_parser

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=100,
            match=match,
            instructions=[],  # DROP
            idle_timeout=BLOCK_IDLE_TIMEOUT,
            hard_timeout=BLOCK_HARD_TIMEOUT
        )

        datapath.send_msg(mod)
        print("🚫 High Severity Device Temporarily Blocked!\n")
