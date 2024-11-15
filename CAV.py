import configparser
import platform
import subprocess
import socket
import time
import json
import ctypes
import logging
import sys
from colorlog import ColoredFormatter
import os
import math
from vnf_generator import VNF
from prometheus_client import start_http_server, Gauge
import tkinter as tk
from PIL import Image, ImageTk
import threading
import queue


class CAV:
    def __init__(self, nodes_to_evaluate=None):
        self.system_os = platform.system()
        self.client_socket = None
        self.connected = False
        self.fec_id = None
        self.user_id = None
        self.my_vnf = None
        self.previous_node = None
        self.next_node = None
        self.next_location = None
        config = configparser.ConfigParser()
        config.read("./ini_files/cav_outdoor.ini")
        self.general = config['general']
        self.conn_status_metric = Gauge('Connection_status', 'State of the CAV connection')
        self.handover_time_metric = Gauge('Handover_time', 'Duration of the handover')
        self.action_time_metric = Gauge('Action_time', 'Time elapsed until an action is received')
        start_http_server(9000)
        self.logger = logging.getLogger('cav')
        self.logger.setLevel(int(self.general['log_level']))
        self.logger.addHandler(logging.FileHandler(self.general['log_file_name'], mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)
        logging.getLogger('pika').setLevel(logging.WARNING)
        if self.general['rover_if'] != 'n' and self.general['rover_if'] != 'N':
            from dronekit import connect, VehicleMode
            self.vehicle = connect(self.general['rover_conn'], wait_ready=True, baud=115200)
            self.logger.info("[I] Connected to vehicle")

            self.vehicle.mode = VehicleMode("GUIDED")
            while not self.vehicle.mode == VehicleMode("GUIDED"):
                time.sleep(1)
            self.logger.info("[I] Guided mode ready")

            self.vehicle.armed = True
            while not self.vehicle.armed:
                time.sleep(1)
            self.logger.info("[I] Armed vehicle")
        else:
            self.vehicle = None
            self.vehicle_active = False

        self.task_queue = queue.Queue()
        self.tkinter_thread = None
        self.start_cav(nodes_to_evaluate)

    def get_data_by_console(self, data_type, message):
        # Function that reads a console entry asking for data to store into a variable
        valid = False
        output = None
        if data_type == int:
            while not valid:
                try:
                    output = int(input(message))
                    valid = True
                except ValueError:
                    self.logger.warning('[!] Error in introduced data! Must use int values. Try again...')
                    valid = False
                except Exception as e:
                    self.logger.warning('[!] Unexpected error ' + str(e) + '! Try again...')
                    valid = False
        elif data_type == float:
            while not valid:
                try:
                    output = float(input(message))
                    valid = True
                except ValueError:
                    self.logger.warning('[!] Error in introduced data! Must use float values. Try again...')
                    valid = False
                except Exception as e:
                    self.logger.warning('[!] Unexpected error ' + str(e) + '! Try again...')
                    valid = False
        else:
            self.logger.error('[!] Data type getter not implemented!')
        return output

    # def get_mac_to_connect(self):
    #     # Function that returns the wireless_conn_manager the mac of the best FEC to connect.
    #     # Takes into account a hysteresis margin of 5 dB for changing FEC
    #     # return 'ab:cd:ef:ac:cd:ef'
    #     best_mac = ''
    #     if self.system_os == 'Windows':
    #         from get_rx_rssi import get_BSSI
    #         json_data = json.loads(str(get_BSSI()).replace("'", "\""))
    #         if len(json_data) > 0:
    #             best_pow = -100
    #             best_val = -1
    #             val = 0
    #             current_pow = -100
    #             while val < len(json_data):
    #                 # self.logger.debug('[D] ' + str(json_data[str(val)]))
    #                 if int(json_data[str(val)][2]) > best_pow:
    #                     best_pow = int(json_data[str(val)][2])
    #                     best_val = val
    #                 if best_mac == json_data[str(val)][0]:
    #                     current_pow = int(json_data[str(val)][2])
    #                 val += 1
    #             if current_pow < int(json_data[str(best_val)][2]) - 5:
    #                 return json_data[str(best_val)][0]
    #             else:
    #                 return best_mac
    #         else:
    #             return best_mac
    #     elif self.system_os == 'Linux':
    #         data = []
    #         try:
    #             iwlist_scan = subprocess.check_output(['sudo', 'iwlist', 'wlan0', 'scan'],
    #                                                   stderr=subprocess.STDOUT)
    #         except subprocess.CalledProcessError as e:
    #             self.logger.error('[!] Unexpected error:' + str(e))
    #         else:
    #             iwlist_scan = iwlist_scan.decode('utf-8').split('Address: ')
    #             i = 1
    #             while i < len(iwlist_scan):
    #                 bssid = iwlist_scan[i].split('\n')[0]
    #                 ssid = iwlist_scan[i].split('ESSID:"')[1].split('"')[0]
    #                 power = iwlist_scan[i].split('level=')[1].split(' dBm')[0]
    #                 if ssid == 'Test301':
    #                     cell = ssid + ' ' + bssid + ' ' + power
    #                     data.append(cell)
    #                 i += 1
    #
    #         if len(data) > 0:
    #             best_pow = -100
    #             val = 0
    #             current_pow = -100
    #             best_pow_mac = ""
    #             while val < len(data):
    #                 # self.logger.debug('[D] ' + data[val])
    #                 split_data = data[val].split(' ')
    #                 i = 0
    #                 while i < len(split_data):
    #                     if split_data[i] == '':
    #                         split_data.pop(i)
    #                     else:
    #                         i += 1
    #                 if split_data[0] != self.general['wifi_ssid']:
    #                     pass
    #                 else:
    #                     if int(split_data[2]) > best_pow:
    #                         best_pow = int(split_data[2])
    #                         best_pow_mac = split_data[1]
    #                     if best_mac == split_data[1]:
    #                         current_pow = int(split_data[2])
    #                 val += 1
    #             if current_pow < best_pow - 5:
    #                 return best_pow_mac
    #             else:
    #                 return best_mac
    #         else:
    #             return best_mac
    #     else:
    #         self.logger.critical('[!] System OS not supported! Please, stop program...')
    #         return

    def get_fec_to_connect(self):
        if (self.my_vnf['current_node'] == 0 or
                self.my_vnf['current_node'] == 1 or
                self.my_vnf['current_node'] == 4 or
                self.my_vnf['current_node'] == 5):
            return 0
        else:
            return 1

    def get_ip_to_connect(self):
        if self.my_vnf['source'] == 0 or self.my_vnf['source'] == 1 or self.my_vnf['source'] == 4 \
                or self.my_vnf['source'] == 5:
            return self.general['fec_0_ip']
        elif self.my_vnf['source'] == 2 or self.my_vnf['source'] == 3 or self.my_vnf['source'] == 6 \
                or self.my_vnf['source'] == 7:
            return self.general['fec_1_ip']
        elif self.my_vnf['source'] == 8 or self.my_vnf['source'] == 9 or self.my_vnf['source'] == 12 \
                or self.my_vnf['source'] == 13:
            return self.general['fec_2_ip']
        elif self.my_vnf['source'] == 10 or self.my_vnf['source'] == 11 or self.my_vnf['source'] == 14 \
                or self.my_vnf['source'] == 15:
            return self.general['fec_3_ip']
        else:
            self.logger.error('[!] Non-existing VNF! Can not choose FEC to connect')

    def handover(self, previous_fec, new_fec):
        # Function that handles handovers. First disconnects from current FEC and after connects to the new one
        self.logger.debug('[D] Performing handover to ' + str(new_fec))
        start_time = time.time()
        self.disconnect(False, previous_fec)
        self.fec_connect(new_fec)
        end_time = time.time()
        self.handover_time_metric.set(end_time - start_time)

    def disconnect(self, starting, fec_id):
        # Disconnects from current FEC
        try:
            if not starting and self.connected:
                message = json.dumps(dict(type="bye"))  # take input
                self.client_socket.send(message.encode())  # send message
                self.client_socket.recv(1024).decode()  # receive response
            if self.general['wifi_if'] == 'y' or self.general['wifi_if'] == 'Y':
                if self.system_os == 'Windows':
                    process_disconnect = subprocess.Popen(
                        'netsh wlan disconnect',
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
                    process_disconnect.communicate()
                    self.connected = False
                elif self.system_os == 'Linux':
                    process_disconnect = subprocess.Popen(
                        'sudo nmcli con down "' + self.general['wifi_ssid'] + ' ' + str(fec_id) + '"',
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
                    process_disconnect.communicate()
                    self.connected = False
                else:
                    self.logger.critical('[!] System OS not supported! Please, stop program...')
                    return
        except ConnectionResetError:
            self.logger.warning('[!] Trying to reuse killed connection!')
        except Exception as e:
            self.logger.exception(e)

    def fec_connect(self, fec_id_or_ip):
        # This function manages connecting to a new FEC given its MAC address
        if self.general['wifi_if'] == 'y' or self.general['wifi_if'] == 'Y':
            if self.system_os == 'Windows':
                while not self.connected:
                    process_connect = subprocess.Popen(
                        self.general['wifi_handler_file'] + ' /ConnectAP "' + self.general[
                            'wifi_ssid'] + '" "' + str(fec_id_or_ip) + '"',
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
                    process_connect.communicate()
                    time.sleep(2)
                    if self.general['wifi_ssid'] in str(subprocess.check_output("netsh wlan show interfaces")):
                        self.logger.debug('[D] Connected!')
                        self.connected = True
                    else:
                        self.logger.warning('[!] Connection not established! Killing query and trying again...')
                        process_connect.kill()
                        process_connect.communicate()
                        time.sleep(1)
            elif self.system_os == 'Linux':
                while not self.connected:
                    try:
                        process_connect = subprocess.Popen(
                            'sudo nmcli connection up "' + str(self.general['wifi_ssid']) +
                            ' ' + str(fec_id_or_ip) + '"',
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                        process_connect.communicate()
                        time.sleep(3)
                        if self.general['wifi_ssid'] in str(subprocess.check_output("iwgetid")):
                            self.logger.debug('[D] Connected!')
                            self.connected = True
                            self.conn_status_metric.set(int(fec_id_or_ip))

                        else:
                            self.logger.warning('[!] Connection not established! Killing query and trying again...')
                            process_connect.kill()
                            process_connect.communicate()
                            time.sleep(1)
                    except subprocess.CalledProcessError:
                        print('Error!')
                        pass
            else:
                self.logger.critical('[!] System OS not supported! Please, stop program...')
                return

            host = self.general['fec_ip']
        else:
            host = fec_id_or_ip
        port = int(self.general['fec_port'])  # socket server port number
        self.client_socket = socket.socket()
        ready = False
        while not ready:
            try:
                self.client_socket.connect((host, port))  # connect to the server
                ready = True
            except OSError:
                time.sleep(1)
        auth_valid = False
        while not auth_valid:
            message = json.dumps(dict(type="auth", user_id=self.user_id))  # take input

            self.client_socket.send(message.encode())  # send message
            data = self.client_socket.recv(1024).decode()  # receive response
            json_data = json.loads(data)
            if json_data['res'] == 200:
                self.logger.debug('[D] Successfully authenticated to FEC ' + str(json_data['id']) + '!')
                self.fec_id = json_data['id']
                auth_valid = True
                if self.my_vnf is not None:
                    self.my_vnf['cav_fec'] = self.fec_id
            else:
                self.logger.error('[!] Error ' + str(json_data['res']) + ' when authenticating to FEC!')
                if self.general['training_if'] != 'y' and self.general['training_if'] != 'Y':
                    self.user_id = self.get_data_by_console(int, '[*] Introduce a valid user ID: ')
                else:
                    self.user_id = 1

    def generate_vnf(self):
        # This function returns a new VNF object whose fields are given by console
        source = self.get_data_by_console(int, '[*] Introduce the source position: ')
        target = self.get_data_by_console(int, '[*] Introduce the target position: ')
        gpu = self.get_data_by_console(int, '[*] Introduce the needed GPU MB: ')
        ram = self.get_data_by_console(int, '[*] Introduce the needed RAM MB: ')
        bw = self.get_data_by_console(int, '[*] Introduce the needed bandwidth (Mbps): ')

        return dict(source=source, target=target, gpu=gpu, ram=ram, bw=bw, previous_node=source,
                    current_node=source, cav_fec=self.fec_id)

    def distance(self, lat1, lng1, lat2, lng2):
        # Finds the distance between two sets of coordinates
        deg_to_rad = math.pi / 180
        d_lat = (lat1 - lat2) * deg_to_rad
        d_lng = (lng1 - lng2) * deg_to_rad
        a = pow(math.sin(d_lat / 2), 2) + math.cos(lat1 * deg_to_rad) * \
            math.cos(lat2 * deg_to_rad) * pow(math.sin(d_lng / 2), 2)
        b = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371000 * b

    def kill_thread(self, thread_id):
        # This functions kills a thread. It is used for stopping the program or disconnecting from a FEC
        try:
            ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit))
            # ref: http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
            if ret == 0:
                raise ValueError("Thread ID " + str(thread_id) + " does not exist!")
            elif ret > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                raise SystemError("PyThreadState_SetAsyncExc failed")
            self.logger.debug("[D] Successfully killed thread " + str(thread_id))
        except Exception as e:
            self.logger.exception(e)

    def stop_program(self):
        self.logger.debug('[!] Ending...')

        if self.system_os == 'Linux' and self.general['video_if'] == 'y' or self.general['video_if'] == 'Y':
            os.system("sudo screen -S ue-stream -X stuff '^C\n'")
        elif self.system_os == 'Windows' and self.general['video_if'] == 'y' or self.general['video_if'] == 'Y':
            os.system("taskkill /im vlc.exe")

        self.disconnect(False, self.my_vnf['cav_fec'])

        if self.system_os == 'Linux' and self.general['wireshark_if'] != 'n' and self.general['wireshark_if'] != 'N':
            os.system("sudo screen -S ue-wireshark -X stuff '^C\n'")
        if self.system_os == 'Linux' and self.general['rover_if'] != 'n' and self.general['rover_if'] != 'N':
            self.logger.debug('[D] Disarming vehicle...')
            self.vehicle.armed = False
            self.vehicle.close()

    def run_tkinter_app(self):
        root = tk.Tk()
        app = StreetGrid(root, self.task_queue, self.my_vnf['source'], self.my_vnf['target'], self.general)
        root.mainloop()

    def start_cav(self, nodes_to_evaluate):
        # Main function
        try:
            # Get user_id
            if self.general['training_if'] != 'y' and self.general['training_if'] != 'Y':
                self.user_id = self.get_data_by_console(int, '[*] Introduce your user ID: ')
            else:
                self.user_id = 1

            if self.system_os == 'Linux':
                wireshark_if = self.general['wireshark_if']
                if wireshark_if != 'n' and wireshark_if != 'N':
                    script_path = os.path.dirname(os.path.realpath(__file__))
                    os.system(
                        "sudo screen -S ue-wireshark -m -d sudo wireshark -i " + self.general[
                            'wlan_if_name'] + " -k -w " +
                        script_path + "/logs/ue-wireshark.pcap")
                video_if = self.general['video_if']
            elif self.system_os == 'Windows':
                video_if = self.general['video_if']
            else:
                self.logger.critical('[!] System OS not supported!')
                exit(-1)

            # In case of being connected to a network, disconnect
            # self.disconnect(True, None)

            # Generate VNF
            stop = False
            if self.general['training_if'] != 'y' and self.general['training_if'] != 'Y':
                self.my_vnf = self.generate_vnf()
            else:
                random_vnf = VNF(nodes_to_evaluate=nodes_to_evaluate, nodes_for_bg_vehicles=None).get_request()
                self.my_vnf = dict(source=random_vnf['source'],
                                   target=random_vnf['target'], gpu=random_vnf['gpu'],
                                   ram=random_vnf['ram'], bw=random_vnf['bw'],
                                   previous_node=random_vnf['source'],
                                   current_node=random_vnf['source'], cav_fec=self.fec_id)

            # Start of the GUI thread
            if self.general['gui_if'] == 'y' or self.general['gui_if'] == 'Y':
                self.tkinter_thread = threading.Thread(target=self.run_tkinter_app)
                self.tkinter_thread.daemon = True 
                self.tkinter_thread.start()


            # Get the best FEC in terms of power and connect to it
            if self.general['wifi_if'] == 'y' or self.general['wifi_if'] == 'Y':
                self.fec_connect(self.get_fec_to_connect())
            else:
                self.fec_connect(self.get_ip_to_connect())

            if self.system_os == 'Linux':
                if video_if == 'y' or video_if == 'Y':
                    os.system("sudo screen -S ue-stream -m -d nvgstplayer-1.0 -i " + str(self.general['video_link']))
            elif self.system_os == 'Windows':
                if video_if == 'y' or video_if == 'Y':
                    os.system("vlc " + self.general['video_link'])		

            try:
                while True:
                    message = json.dumps(dict(type="vnf", user_id=self.user_id, data=self.my_vnf))  # take input
                    self.client_socket.send(message.encode())  # send message
                    data = self.client_socket.recv(1024).decode()  # receive response
                    json_data = json.loads(data)
                    self.logger.debug('[D] Response from server: ' + str(json_data))
                    if json_data['res'] == 200:
                        self.next_node = json_data['next_node']
                        if self.vehicle is not None and self.next_node != -1:
                            self.next_location = json_data['location']
                        if json_data['next_node'] == -1:
                            self.logger.debug('[D] Car reached target!')
                            if self.general['training_if'] != 'y' and self.general['training_if'] != 'Y':
                                key_in = input('[?] Want to send a new VNF? Y/n: (Y) ')
                            else:
                                key_in = 'n'
                            if key_in != 'n':
                                self.my_vnf = None
                                stop = False
                            else:
                                self.my_vnf = None
                                stop = True
                        else:
                            stop = False
                    elif json_data['res'] == 403:
                        self.my_vnf = None
                        self.logger.error('[!] Error! Required resources are not available on current FEC. '
                                          'Ask for less resources.')
                        stop = True
                    elif json_data['res'] == 404:
                        self.logger.error('[!] Error! Required target does not exist. Ask for an existing target. '
                                          'my_vnf: ' + str(self.my_vnf))
                        self.my_vnf = None
                        stop = True
                    else:
                        self.my_vnf = None
                        self.logger.error('[!] Error ' + str(json_data['res']) + ' when sending VNF to FEC!')
                        stop = True

                    while self.my_vnf is not None:
                        # Move to next point
                        if self.general['wifi_if'] == 'y' or self.general['wifi_if'] == 'Y':
                            if json_data['cav_fec'] is not self.my_vnf['cav_fec']:
                                self.handover(self.my_vnf['cav_fec'], json_data['cav_fec'])
                        else:
                            if json_data['cav_fec'] is not self.my_vnf['cav_fec']:
                                self.handover(None, json_data['fec_ip'])

                        # Tkinter to moving state

                        self.task_queue.put({"command": "transit", "curr": self.my_vnf["current_node"], "next": self.next_node})

                        if self.vehicle is not None and self.vehicle_active is False:
                            import dronekit
                            point = dronekit.LocationGlobal(float(self.next_location.split(',')[0]),
                                                            float(self.next_location.split(',')[1]), 0)
                            self.logger.debug('[D] Moving towards first target...')
                            self.vehicle.simple_goto(point, 1)
                            self.vehicle_active = True
                        if self.vehicle is not None and self.vehicle_active is True:
                            while self.distance(float(self.next_location.split(',')[0]),
                                                float(self.next_location.split(',')[1]),
                                                self.vehicle.location.global_frame.lat,
                                                self.vehicle.location.global_frame.lon) > 3:
                                time.sleep(1)
                        else:
                            if self.general['training_if'] != 'y' and self.general['training_if'] != 'Y':
                                input('[*] Press Enter when getting to the next point...')

                        # Update state vector
                        self.logger.debug('[D] Reaching next point! Sending changes to FEC...')
                        self.my_vnf['previous_node'] = self.my_vnf['current_node']
                        self.my_vnf['current_node'] = self.next_node
                        self.my_vnf['cav_fec'] = self.fec_id
                        message = json.dumps(dict(type="state", user_id=self.user_id,
                                                  data=dict(previous_node=self.my_vnf['previous_node'],
                                                            current_node=self.my_vnf['current_node'],
                                                            cav_fec=self.my_vnf['cav_fec'])))
                        self.task_queue.put({"command": "arrived", "dest": self.my_vnf["current_node"]})
                        sent_time = time.time()
                        self.client_socket.send(message.encode())  # send message
                        data = self.client_socket.recv(1024).decode()  # receive response
                        final_time = time.time()
                        self.action_time_metric.set((final_time - sent_time) * 1000)
                        json_data = json.loads(data)
                        self.logger.debug('[D] Response from server: ' + str(json_data))
                        if json_data['res'] == 200:
                            self.next_node = json_data['next_node']
                            if self.vehicle is not None and json_data['next_node'] != -1:
                                arriving_location = self.next_location
                                self.next_location = json_data['location']
                            if json_data['next_node'] == -1:
                                self.logger.debug('[D] Car reached target!')
                                self.conn_status_metric.set(-1)
                                if self.general['training_if'] != 'y' and self.general['training_if'] != 'Y':
                                    key_in = input('[?] Want to send a new VNF? Y/n: (Y) ')
                                else:
                                    key_in = 'n'
                                if key_in != 'n' and key_in != 'N':
                                    self.my_vnf = None
                                    stop = False
                                else:
                                    self.my_vnf = None
                                    self.task_queue.put({"command": "exit"})
                                    stop = True
                            else:
                                stop = False
                                if self.vehicle is not None and self.vehicle_active is True:
                                    while self.distance(float(arriving_location.split(',')[0]),
                                                        float(arriving_location.split(',')[1]),
                                                        self.vehicle.location.global_frame.lat,
                                                        self.vehicle.location.global_frame.lon) > 1:
                                        time.sleep(1)
                                    self.logger.debug('[D] Reached next point! Loading next target...')
                                    point = dronekit.LocationGlobal(float(self.next_location.split(',')[0]),
                                                                    float(self.next_location.split(',')[1]), 0)
                                    self.vehicle.simple_goto(point, 1)
                        elif json_data['res'] == 403:
                            self.my_vnf = None
                            self.logger.error('[!] Error! Required resources are not available on current FEC. '
                                              'Ask for less resources.')
                            stop = True
                        elif json_data['res'] == 404:
                            self.my_vnf = None
                            self.logger.error('[!] Error! Required target does not exist. Ask for an existing target.')
                            stop = True
                        else:
                            self.my_vnf = None
                            self.logger.error('[!] Error ' + str(json_data['res']) + ' when sending VNF to FEC!')
                            stop = True
                    if stop:
                        break
                self.disconnect(False, self.fec_id)

            except ConnectionRefusedError:
                self.logger.error('[!] FEC server not available! Please, press enter to stop client.')
            except SystemExit:
                message = json.dumps(dict(type="bye"))  # take input
                self.client_socket.send(message.encode())  # send message
                self.client_socket.close()  # close the connection
            except Exception as e:
                self.logger.exception(e)
        except KeyboardInterrupt:
            self.stop_program()
        except Exception as e:
            self.logger.exception(e)
            self.stop_program()


class StreetGrid:
    def __init__(self, master, task_queue, source, destination, general):
        self.master = master
        self.task_queue = task_queue
        self.master.title("StreetGrid")	
        self.scenario_img_path = "scenario4.png"
        self.car_img_path = "car.png"
        self.flag_img_path = "flag.png"
        self.canvas = tk.Canvas(master, width=1200, height=500)
        self.canvas.pack()
        self.general = general
        self.load_background_img()
        self.load_car_img(source)
        self.load_flag_img(destination)

        self.check_queue()

    def load_background_img(self):
        self.bg_image = Image.open(self.scenario_img_path)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)

    def load_car_img(self, node):
        pos = self.general[('img_node_' + str(node))].split(',')

        x = int(pos[0])
        y = int(pos[1])
        self.car_image = Image.open(self.car_img_path)
        self.car_image = self.car_image.resize((100, 100))
        self.car_photo = ImageTk.PhotoImage(self.car_image)
        self.car = self.canvas.create_image(x, y, anchor="center", image=self.car_photo)

    def load_flag_img(self, node):
        pos = self.general[('img_node_' + str(node))].split(',')
        x = int(pos[0])
        y = int(pos[1])
        self.flag_image = Image.open(self.flag_img_path)
        self.flag_image = self.flag_image.resize((50, 50))
        self.flag_photo = ImageTk.PhotoImage(self.flag_image)
        self.flag = self.canvas.create_image(x + 30, y + 50, anchor="center", image=self.flag_photo)


    def set_car_angle(self, angle):
        self.car_image = Image.open(self.car_img_path)
        self.car_image = self.car_image.resize((100, 100))
        self.car_image = self.car_image.rotate(angle, expand=True)
        self.car_photo = ImageTk.PhotoImage(self.car_image)
        self.canvas.itemconfig(self.car, image=self.car_photo)

    def car_in_transit(self, prev_node, next_node):
        prev_pos = self.general[('img_node_' + str(prev_node))].split(',')
        next_pos = self.general[('img_node_' + str(next_node))].split(',')

        if int(prev_pos[0]) > int(next_pos[0]):
            self.set_car_angle(90)
        elif int(prev_pos[0]) < int(next_pos[0]):
            self.set_car_angle(-90)
        if int(prev_pos[1]) > int(next_pos[1]):
            self.set_car_angle(0)
        elif int(prev_pos[1]) < int(next_pos[1]):
            self.set_car_angle(180)
        x = (int(prev_pos[0]) + int(next_pos[0])) / 2
        y = (int(prev_pos[1]) + int(next_pos[1])) / 2
        self.canvas.coords(self.car, x, y)

    def car_arrived(self, dest_node):
        dest_pos = self.general[('img_node_' + str(dest_node))].split(',')
        x = int(dest_pos[0])
        y = int(dest_pos[1])
        self.canvas.coords(self.car, x, y)


    def check_queue(self):
        while not self.task_queue.empty():
            task = self.task_queue.get()
            if task['command'] == "transit":
                current_node = task['curr']
                next_node = task['next']
                self.car_in_transit(current_node, next_node)
            elif task['command'] == "arrived":
                dest_node = task['dest']
                self.car_arrived(dest_node)
            elif task['command'] == "exit":
                self.master.quit()
        self.master.after(100, self.check_queue)


if __name__ == '__main__':
    my_cav = CAV()