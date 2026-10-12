import socket
import psutil

def get_lan_network_info():
    """
    ค้นหา IP และ MAC address จาก Interface พอร์ต LAN (เช่น eth0, enp3s0, eno1)
    """
    interfaces = psutil.net_if_addrs()
    
    # ลำดับความสำคัญของ prefix ชื่ออินเทอร์เฟซพอร์ต LAN
    lan_prefixes = ('eth', 'en')
    
    selected_ip = None
    selected_mac = None

    # วนลูปหาอินเทอร์เฟซที่ขึ้นต้นด้วย eth หรือ en
    for iface_name, addrs in interfaces.items():
        if any(iface_name.startswith(prefix) for prefix in lan_prefixes):
            ip = None
            mac = None
            for addr in addrs:
                # ดึง IPv4 (AF_INET)
                if addr.family == socket.AF_INET:
                    ip = addr.address
                # ดึง MAC Address (AF_PACKET บน Linux)
                elif hasattr(socket, 'AF_PACKET') and addr.family == socket.AF_PACKET:
                    mac = addr.address

            # ถ้าพบ IP ที่ไม่ใช่ loopback หรือค่าว่าง ให้เลือก interface นี้
            if ip and not ip.startswith('127.'):
                selected_ip = ip
                selected_mac = mac
                break
            # สำรองไว้ถ้าเจอ mac แต่ ip ยังไม่ได้เซ็ต
            elif mac and not selected_mac:
                selected_mac = mac

    return selected_ip or "127.0.0.1", selected_mac or "00:00:00:00:00:00"