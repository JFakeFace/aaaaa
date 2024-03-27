# Import python libraries
import time
import math
import copy
import queue
import argparse
import random as rnd
import subprocess
from threading import Thread

import numpy as np
from matplotlib import pyplot as plt

# Import RapidChiplet files
import helpers as hlp
import validation as vld


name_of_design = None
priority_map = {0:1,1:2,2:2,3:1,4:2}
# Computes a full source-destination path for each combination of sending and receiving chiplets in the following
def compute_area_summary(chiplets, placement):
    total_chiplet_area = 0
    # Smallest and largest coordinates occupied by a chiplet
    (minx, miny, maxx, maxy) = (float("inf"), float("inf"), -float("inf"), -float("inf"))
    # Iterate through chiplets
    for chiplet_desc in placement["chiplets"]:
        chiplet = chiplets[chiplet_desc["name"]]
        (x, y) = (chiplet_desc["position"]["x"], chiplet_desc["position"]["y"])  # Position
        (w, h) = (chiplet["dimensions"]["x"], chiplet["dimensions"]["y"])  # Dimensions
        # Add this chiplet's are to total area
        total_chiplet_area += (w * h)
        # Update min and max coordinates
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x + w)
        maxy = max(maxy, y + h)
    # Compute total interposer area
    chip_width = (maxx - minx)
    chip_height = (maxy - miny)
    total_interposer_area = chip_width * chip_height
    area_summary = {
        "chip_width": chip_width,
        "chip_height": chip_height,
        "total_chiplet_area": total_chiplet_area,
        "total_interposer_area": total_interposer_area
    }
    return area_summary

# traffic classes: core->core, core->memory, core->io, memory->io

def construct_ici_routing(ici_graph, routing):
    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    # Compute a routing per traffic-class.
    classes = ["C2C", "C2M", "C2I", "M2I"]
    # The following two dictionaries are the result of this function - they fully determine the routing
    paths_per_class = {cls: {} for cls in classes}
    n_paths_per_edge_per_class = {cls: {(src, dst): 0 for src in range(n) for dst in neighbors[src]} for cls in classes}
    # Cover all traffic classes without running Dijkstra twice on the same start-vertex
    src_types = ["C", "M"]
    dst_types_by_src_type = {"C": ["C", "M", "I"], "M": ["I"]}
    for src_type in src_types:
        # Run Dijkstra for each sending node in a given traffic class
        # We minimize the number of hops, not the latency.
        for src in nodes_by_type[src_type]:
            dist = [float("inf") for i in range(n)]  # Distance from SRC in hops
            preds = [[] for i in range(n)]  # Predecessors (can be many for multiple shortest paths)
            todo = queue.PriorityQueue()  # Visited but not yet processed nodes
            dist[src] = 0
            todo.put((0, src))
            # Explore paths from src to all chiplets
            while todo.qsize() > 0:
                (cur_dist, cur) = todo.get()
                # A shorter path to the cur-node has been found -> skip
                if cur_dist > dist[cur]:
                    continue
                # Iterate through neighbors of the cur-node
                for nei in neighbors[cur]:
                    nei_dist = cur_dist + 1
                    # We found a path to nei that is shorter than the currently best known one
                    if nei_dist < dist[nei]:
                        dist[nei] = nei_dist
                        preds[nei] = [cur]
                        # Only enqueue the "nei"-node for processing if it can relay traffic
                        if relay_map[nei]:
                            todo.put((nei_dist, nei))
                    # We found a path equally shorter than the shortest path
                    elif (routing in ["random", "balanced"]) and (nei_dist == dist[nei]) and (cur not in preds[nei]):
                        preds[nei].append(cur)
            # Use backtracking to construct all src->dst paths for the given traffic class
            for dst_type in dst_types_by_src_type[src_type]:
                for dst in nodes_by_type[dst_type]:
                    cls = src_type + "2" + dst_type
                    # Only look at paths with at least one hop
                    if dst == src:
                        continue
                    path = [dst]
                    cur = dst
                    while cur != src:
                        # Balance paths across links
                        if routing == "balanced":
                            n_paths = [n_paths_per_edge_per_class[cls][(pred, cur)] for pred in preds[cur]]
                            pred = preds[cur][n_paths.index(min(n_paths))]
                        # Randomly select shortest paths
                        
                        elif routing == "random":
                            pred = preds[cur][rnd.randint(0, len(preds[cur]) - 1)]
                        # Use the minimum index (what BookSim does)
                        else:
                            pred = preds[cur][0]
                        n_paths_per_edge_per_class[cls][(pred, cur)] += 1
                        cur = pred
                        path.insert(0, cur)
                    paths_per_class[cls][(src, dst)] = path
    # Return results
    return (paths_per_class, n_paths_per_edge_per_class)
# Compute all link lengths
def compute_link_summary(chiplets, placement, topology, packaging):
    link_lengths = []
    link_lengths_internal = {}
    for link in topology:
        endpoints = [link["ep1"], link["ep2"]]
        # Compute positions of start-and endpoint
        positions = []
        node_ids = []
        for endpoint in endpoints:
            if endpoint["type"] == "chiplet":
                chiplet_desc = placement["chiplets"][endpoint["outer_id"]]
                chiplet = chiplets[chiplet_desc["name"]]
                # Rotate the chiplet if needed
                chiplet = hlp.rotate_chiplet(chiplet, chiplet_desc["rotation"])
                phy = chiplet["phys"][endpoint["inner_id"]]
                positions.append((chiplet_desc["position"]["x"] + phy["x"], chiplet_desc["position"]["y"] + phy["y"]))
                node_ids.append(endpoint["outer_id"])
            else:
                irouter = placement["interposer_routers"][endpoint["outer_id"]]
                positions.append((irouter["position"]["x"], irouter["position"]["y"]))
                node_ids.append(len(placement["chiplets"]) + endpoint["outer_id"])
        # Compute link length
        if packaging["link_routing"] == "manhattan":
            length = sum([abs(positions[0][dim] - positions[1][dim]) for dim in range(2)])
            link_lengths.append(length)
            link_lengths_internal[tuple(node_ids)] = length
            link_lengths_internal[tuple(reversed(node_ids))] = length
        elif packaging["link_routing"] == "euclidean":
            # todo: fault about euclidean
            length = math.sqrt(sum([abs(positions[0][dim] - positions[1][dim]) for dim in range(2)]))
            link_lengths.append(length)
            link_lengths_internal[tuple(node_ids)] = length
            link_lengths_internal[tuple(reversed(node_ids))] = length
    # Summarize link lengths
    link_summary = {
        "avg": sum(link_lengths) / len(link_lengths),
        "min": min(link_lengths),
        "max": max(link_lengths),
        "all": link_lengths
    }
    return (link_summary, link_lengths_internal)
def compute_manufacturing_cost(technology, chiplets, placement, packaging, area_summary):
    # First, compute the manufacturing cost per chiplet
    results_per_chiplet = {}
    for chiplet_name in set([x["name"] for x in placement["chiplets"]]):
        results_per_chiplet[chiplet_name] = {}
        chiplet = chiplets[chiplet_name]
        tech = technology[chiplet["technology"]]
        wr = tech["wafer_radius"]  # Wafer radius
        dd = tech["defect_density"]  # Defect density
        wc = tech["wafer_cost"]  # Wafer cost
        ca = chiplet["dimensions"]["x"] * chiplet["dimensions"]["y"]  # Chiplet area
        # Dies per wafer
        dies_per_wafer = int(math.floor(((math.pi * wr ** 2) / ca) - ((math.pi * 2 * wr) / math.sqrt(2 * ca))))
        results_per_chiplet[chiplet_name]["dies_per_wafer"] = dies_per_wafer
        # Manufacturing yield
        manufacturing_yield = 1.0 / (1.0 + dd * ca)
        results_per_chiplet[chiplet_name]["manufacturing_yield"] = manufacturing_yield
        # Known good dies
        known_good_dies = dies_per_wafer * manufacturing_yield
        results_per_chiplet[chiplet_name]["known_good_dies"] = known_good_dies
        # Cost
        cost = wc / known_good_dies
        results_per_chiplet[chiplet_name]["cost"] = cost
    # Next, compute the manufacturing cost of the interposer if an interposer is used
    results_interposer = {"cost": 0}
    if packaging["has_interposer"]:
        ip_tech = technology[packaging["interposer_technology"]]
        wr = ip_tech["wafer_radius"]  # Wafer radius
        dd = ip_tech["defect_density"]  # Defect density
        wc = ip_tech["wafer_cost"]  # Wafer cost
        ia = area_summary["total_interposer_area"]  # Interposer area
        # Dies per wafer
        dies_per_wafer = int(math.floor(((math.pi * wr ** 2) / ia) - ((math.pi * 2 * wr) / math.sqrt(2 * ia))))
        results_interposer["dies_per_wafer"] = dies_per_wafer
        # Manufacturing yield
        manufacturing_yield = 1.0 / (1.0 + dd * ia)
        results_interposer["manufacturing_yield"] = manufacturing_yield
        # Known good dies
        known_good_dies = dies_per_wafer * manufacturing_yield
        results_interposer["known_good_dies"] = known_good_dies
        # Cost
        cost = wc / known_good_dies
        results_interposer["cost"] = cost
    # Compute the overall cost per working chip
    py = packaging["packaging_yield"]  # Packaging yield
    total_cost = (sum([results_per_chiplet[x["name"]]["cost"] for x in placement["chiplets"]]) + results_interposer[
        "cost"]) / py
    return {"total_cost": total_cost, "interposer": results_interposer, "chiplets": results_per_chiplet}


# author: jiaxuming
# time : 2024/2/5 18:42
# chiplet 实例化
###################################################################
event_queue = None
chiplet_ins = None
waiting = 0

class message:

    def __init__(self, data_id=-1, data_src=-1, data_end=-1, next_chiplet=-1,
                 data_type=-1, data_size=0, data_time=0, src_port=-1, end_port=-1,old_port = None,priority = 1):
        self.data_id = data_id
        self.data_src = data_src
        self.data_end = data_end
        self.next_chiplet = next_chiplet
        self.data_type = data_type  # data 0 表示 credit确认消息 1 表示路由消息 2 表示端口询问input缓冲区 3 询问输出缓冲区是否有位置
        self.data_size = data_size
        self.data_time = data_time  # 数据到达时间
        self.src_port = src_port
        self.end_port = end_port
        self.old_port = old_port
        self.priority = priority

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"data_id={self.data_id}, "
                f"data_src={self.data_src}, "
                f"data_end={self.data_end}, "
                f"next_chiplet={self.next_chiplet}, "
                f"data_type={self.data_type}, "
                f"data_size={self.data_size}, "
                f"data_time={self.data_time}, "
                f"src_port={self.src_port}, "
                f"end_port={self.end_port}, "
                f"priority={self.priority})")

    def get_data_id(self):
        return self.data_id

    def get_data_end(self):
        return self.data_end

    def get_data_src(self):
        return self.data_src

    def get_data_type(self):
        return self.data_type

    def __lt__(self, other):
        if self.data_time == other.data_time:
            return self.priority < other.priority
        return self.data_time < other.data_time


class chiplet_instance:
    def __init__(self, chiplet_id=-1, port_num=0, input_buffer=0, output_buffer=0):
        self.chiplet_id = chiplet_id
        self.port_num = port_num
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.input_data = [[] for i in range(port_num)]
        self.output_data = [[] for i in range(port_num)]

    # data_id = -1, data_src = -1, data_end = -1, next_chiplet = -1, data_type = -1, data_size = 0, data_time = 0, src_port = -1, end_port = -1


def creat_event1(message, router_map):
    global event_queue
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    # 解析消息中的必要信息
    src_chiplet = message.data_src
    src_port = message.src_port
    dst_chiplet = message.data_end
    next_chiplet = message.next_chiplet
    message.data_time += node_internal_latency[src_chiplet]
    message.data_type = 1
    message.priority = priority_map[message.data_type]
    found = False
    for i in range(len(chiplet_ins[src_chiplet].input_data[message.old_port])):

        if chiplet_ins[src_chiplet].input_data[message.old_port][i].data_id == message.data_id:
            # print(i)
            # print(chiplet_ins[src_chiplet].input_data[src_port][i])
            del chiplet_ins[src_chiplet].input_data[message.old_port][i]
            found = True
            break
    if found == False:
        print('Could not find the data with data_id input: %d' % message.data_id)

    event_queue.put(message)


def ask_input_room(message, router_map, phy_neibors):
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    src_chiplet = message.data_src
    dst_chiplet = message.data_end
    next_chiplet = message.next_chiplet
    port = message.end_port
    if chiplet_ins[next_chiplet].input_buffer > len(chiplet_ins[next_chiplet].input_data[port]):
        # 有空间，发送credit消息
        message_add = copy.deepcopy(message)
        message_add.data_type = 3
        message_add.priority = priority_map[message_add.data_type]
        message_add.data_src = next_chiplet
        if next_chiplet == dst_chiplet:
            message_add.next_chiplet = dst_chiplet
            message_add.src_port = message.end_port
            message_add.end_port = message.end_port
        else:
            message_add.next_chiplet = path_src_dst[(next_chiplet, dst_chiplet)][1]
            message_add.src_port = phy_neibors[(message_add.data_src, message_add.next_chiplet)][0]
            message_add.end_port = phy_neibors[(message_add.data_src, message_add.next_chiplet)][1]

        message_add.data_time += get_path_delay(src_chiplet, next_chiplet, router_map) * 2 + phy_latency[next_chiplet]
        chiplet_ins[next_chiplet].input_data[port].append(message_add)
        send_credit(data_id=message.data_id, src_chiplet=next_chiplet, dst_chiplet=src_chiplet,
                    port_src=message.end_port, port_end=message.src_port, time=message.data_time + get_path_delay(src_chiplet,next_chiplet,router_map))
        message_add.old_port = port
        event_queue.put(message_add)
    else:
        # 没有空间，进行轮询，时间+1
        retry_data_request(message, router_map)


def handle_data_request(message, router_map, phy_neibors):
    global event_queue, chiplet_ins
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    # 解析消息中的必要信息
    src_chiplet = message.data_src
    dst_chiplet = message.data_end
    next_chiplet = message.next_chiplet
    port = message.end_port  # 假设这是下一跳需要检查的端口

    # 计算到达时间，加上路径延迟
    arrival_time = message.data_time + get_path_delay(src_chiplet, next_chiplet, router_map)
    message.data_time = arrival_time

    message.data_type = 4
    message.priority = priority_map[message.data_type]
    event_queue.put(message)


def get_path_delay(src_chiplet, next_chiplet, router_map):
    # 根据router_map获取路径延迟，这里是一个示例实现
    #todo think about it
    # print(src_chiplet,next_chiplet)
    if src_chiplet == next_chiplet:
        return 0
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    return edge_latencies[src_chiplet][next_chiplet]


def send_credit(data_id, src_chiplet, dst_chiplet, port_src, port_end, time):
    # 构建并发送credit消息
    global event_queue
    credit_msg = message(
        data_type=0,  # credit消息类型为0
        priority=priority_map[0],
        data_src=src_chiplet,
        data_end=dst_chiplet,
        data_time=time,
        src_port=port_src,
        end_port=port_end,
        data_id=data_id
    )
    event_queue.put(credit_msg)


def retry_data_request(message, router_map):
    global waiting
    # 增加时间后重新放入事件队列进行轮询
    message.priority += 1
    message.data_time += message.priority  # 时间+1
    waiting += 1
    event_queue.put(message)


# 零事件之后的操作
def release_output_buffer(top_event, router_map, phy_neighbors):
    global event_queue, chiplet_ins
    # 从top_event获取必要的信息
    data_id = top_event.data_id
    found = False
    chiplet_src = top_event.data_src
    chiplet_end = top_event.data_end
    port_src = top_event.src_port
    port_end = top_event.end_port
    # 遍历所有chiplet实例中的输出数据

    for i, message in enumerate(chiplet_ins[chiplet_end].output_data[port_end]):
        if message.data_id == data_id:
            # 找到了需要处理的消息
            # 创建新的消息对象，这里我们直接修改top_event作为示例
            # 实际上，您可能需要根据需要创建一个全新的消息对象
            # message_new = new_message_by_credit(message, router_map, top_event.data_time, phy_neighbors)

            # 将新消息放入全局事件队列
            # event_queue.put(message_new)
            # 从输出数据中删除找到的消息
            del chiplet_ins[chiplet_end].output_data[port_end][i]
            found = True
            break  # 找到后即退出循环
    if not found:
        print(f'Could not find the data with data_id: {data_id}')


def new_message_by_credit(message_credit, router_map, credit_time, phy_neighbors):
    global chiplet_ins
    # global data_id
    # 解包router_map中的各个组件
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    dst_chiplet = message_credit.data_end
    src_chiplet = message_credit.data_src
    next_chiplet = message_credit.next_chiplet

    # 使用新的方法来获取当前芯片到下一个芯片的边缘延迟
    # edge_latencies是一个列表，每个元素是一个字典，表示当前节点到其邻居的延迟
    #todo think about it
    latency_to_next_chiplet = edge_latencies[src_chiplet][next_chiplet] if src_chiplet != next_chiplet else 0

    # 更新数据时间，包括边缘延迟和物理延迟
    data_time = credit_time + latency_to_next_chiplet + phy_latency[next_chiplet]
    # print("new_message_credit")
    port_id = None
    if next_chiplet == dst_chiplet:
        port_id = (message_credit.end_port,message_credit.end_port)
    else:
        port_id = phy_neighbors[(next_chiplet, path_src_dst[(next_chiplet, dst_chiplet)][1])]
    # 创建新的消息对象，完成data_time参数的填充
    message_new = message(data_id=message_credit.data_id, data_src=next_chiplet, data_end=dst_chiplet,
                          next_chiplet=path_src_dst[(next_chiplet, dst_chiplet)][1]
                          if next_chiplet != dst_chiplet else dst_chiplet,
                          data_type=3,
                          priority=priority_map[3],
                          data_size=message_credit.data_size,
                          data_time=data_time,
                          src_port=port_id[0],
                          end_port=port_id[1])
    # todo error
    # 这里可以添加代码，例如将message_new添加到输出缓冲区等
    return message_new

    # 注意：这段代码可能需要根据您的具体实现进行调整


# 当数据进入缓冲区后，向下一跳发送请求信息
def creat_quest_message(message_output, router_map):
    # 解包router_map中的各个组件
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map

    # 从message_output中获取必要的信息
    src_chiplet = message_output.data_src
    next_chiplet = message_output.next_chiplet
    end_chiplet = message_output.data_end
    data_time = message_output.data_time + edge_latencies[src_chiplet][next_chiplet]

    # 假设消息类型为2表示准备好接收信息的消息
    # 注意：这里的message构造函数参数应该与您定义的message类匹配
    quest_message = message(data_id=message_output.data_id, data_src=src_chiplet, data_end=end_chiplet,
                            next_chiplet=next_chiplet, data_type=2, data_size=message_output.data_size,
                            data_time=data_time, src_port=message_output.src_port, end_port=message_output.end_port,priority=priority_map[2])

    return quest_message


def ask_output_buffer(message_output, router_map):
    global chiplet_ins, event_queue,waiting
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    src_chiplet = message_output.data_src
    src_port = message_output.src_port  # 获取源端口号

    # 检查输出缓冲区是否有足够的空间
    if len(chiplet_ins[src_chiplet].output_data[src_port]) < chiplet_ins[src_chiplet].output_buffer:
        # 如果有足够的空间，将消息放入src_chiplet的输入缓冲区
        chiplet_ins[src_chiplet].output_data[src_port].append(message_output)
        # print('*******************************************')
        # for x in chiplet_ins[src_chiplet].output_data[src_port]:
        #     print(x)
        # print('*******************************************')
        # 假设每个chiplet_instance有一个input_buffer_count属性来跟踪输入缓冲区中的消息数量
        # 如果有足够的空间，创建新的消息，data_type为2
        new_message = message(
            data_id=message_output.data_id,
            data_src=message_output.data_src,
            data_end=message_output.data_end,
            next_chiplet=message_output.next_chiplet,
            data_type=2,  # 设置新消息的类型为2
            data_size=message_output.data_size,
            data_time=message_output.data_time + phy_latency[src_chiplet],  # 增加物理延迟
            src_port=message_output.src_port,
            end_port=message_output.end_port,
            priority=priority_map[2]
        )
        event_queue.put(new_message)  # 将新消息加入到事件队列中

    else:
        # 如果没有足够的空间，将data_time增加1后，把消息放回event_queue中
        message_output.priority += 1
        message_output.data_time += message_output.priority
        waiting += 1
        event_queue.put(message_output)

# Constructs a graph where nodes are chiplets and interposer-routers and edges are links.
def construct_ici_graph(chiplets, placement, topology):
    c = len(placement["chiplets"])  # Number of chiplets
    r = len(placement["interposer_routers"])  # Number of interposer-routers
    n = c + r  # Number of nodes in the graph
    # Construct adjacency list
    neighbors = [[] for i in range(n)]
    phy_neighbors = {}
    # Iterate through links
    for link in topology:
        nid1 = (c if link["ep1"]["type"] == "irouter" else 0) + link["ep1"]["outer_id"]
        nid2 = (c if link["ep2"]["type"] == "irouter" else 0) + link["ep2"]["outer_id"]
        phy_nid1 = link["ep1"]["inner_id"]
        phy_nid2 = link["ep2"]["inner_id"]
        phy_neighbors[(nid1, nid2)] = (phy_nid1, phy_nid2)
        phy_neighbors[(nid2, nid1)] = (phy_nid2, phy_nid1)
        # print(nid1,nid2)
        neighbors[nid1].append(nid2)
        neighbors[nid2].append(nid1)
    # Collect node attributes...
    relay_map = [None for i in range(n)]
    nodes_by_type = {"C": [], "M": [], "I": []}
    # ... for chiplets
    for nid in range(c):
        chiplet = chiplets[placement["chiplets"][nid]["name"]]
        typ = chiplet["type"][0].upper()
        relay_map[nid] = chiplet["relay"]
        nodes_by_type[typ].append(nid)
    # ... for interposer-routers
    for nid in range(c, c + r):
        relay_map[nid] = True
    # Return graph
    return (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors)

# author : jiaxuming
# time : 2024/2/4/19:14
# this function is to compute the router_map
def compute_router_map(technology, chiplets, placement, packaging, ici_graph, ici_routing, link_latencies_internal,routing):
    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    (paths_per_class, n_paths_per_edge_per_class) = ici_routing
    phy_latency = [0 for i in range(n)]
    node_relay_latencies = [(packaging["latency_irouter"] if i >= c else 0) for i in range(n)]
    node_internal_latency = [(packaging["latency_irouter"] if i >= c else 0) for i in range(n)]
    node_latencies = [None for i in range(c)]
    edge_latencies = [{nei: (packaging["link_latency"]
                             if packaging["link_latency_type"] == "constant"
                             else int(math.ceil(eval(packaging["link_latency"])(link_latencies_internal[(i, nei)]))))
                       for nei in neighbors[i]} for i in range(n)]
    for i in range(len(placement["chiplets"])):
        chiplet = chiplets[placement["chiplets"][i]["name"]]
        internal_latency = chiplet["internal_latency"]
        phy_latency[i] = technology[chiplet["technology"]]["phy_latency"]
        node_relay_latencies[i] = internal_latency + 2 * phy_latency[i]
        node_latencies[i] = internal_latency + phy_latency[i]
        node_internal_latency[i] = internal_latency
    # all code up is to compute the latency,and then i will compute the route_map
    src_types = ["C", "M"]
    # todo: I update it
    dst_types_by_src_type = {"C": ["C", "M", "I"], "M": ["I", "C"]}
    path_src_dst = {}
    dist_src_to_any = {}
    for src_type in src_types:
        # Run Dijkstra for each sending node in a given traffic class
        # We minimize the number of hops, not the latency.
        # print(nodes_by_type, src_type)
        for src in nodes_by_type[src_type]:
            dist = [float("inf") for i in range(n)]  # Distance from SRC in hops
            preds = [[] for i in range(n)]  # Predecessors (can be many for multiple shortest paths)
            todo = queue.PriorityQueue()  # Visited but not yet processed nodes
            dist[src] = 0
            todo.put((0, src))
            # Explore paths from src to all chiplets
            while todo.qsize() > 0:
                (cur_dist, cur) = todo.get()
                # A shorter path to the cur-node has been found -> skip
                if cur_dist > dist[cur]:
                    continue
                chiplet_cur = chiplets[placement["chiplets"][cur]["name"]]
                # Iterate through neighbors of the cur-node
                for nei in neighbors[cur]:
                    chiplet_nei = chiplets[placement["chiplets"][nei]["name"]]
                    len_dist = edge_latencies[cur][nei]
                    if chiplet_nei['type'][0] not in ['i', 'c', 'm']:
                        len_dist += packaging["latency_irouter"]
                    else:
                        len_dist += chiplet_nei['internal_latency'] + technology[chiplet_nei["technology"]][
                            "phy_latency"]
                    if chiplet_cur['type'][0] in ['i', 'c', 'm']:
                        len_dist += technology[chiplet_cur["technology"]]["phy_latency"]
                    if len_dist + cur_dist < dist[nei]:
                        dist[nei] = len_dist + cur_dist
                        preds[nei] = [cur]
                        # Only enqueue the "nei"-node for processing if it can relay traffic
                        if relay_map[nei]:
                            todo.put((len_dist + cur_dist, nei))
                    # We found a path equally shorter than the shortest path
                    elif (routing in ["random", "balanced"]) and (len_dist == dist[nei]) and (cur not in preds[nei]):
                        preds[nei].append(cur)

            dist_src_to_any[src] = dist
            # Use backtracking to construct all src->dst paths for the given traffic class
            for dst_type in dst_types_by_src_type[src_type]:
                for dst in nodes_by_type[dst_type]:
                    cls = src_type + "2" + dst_type
                    # Only look at paths with at least one hop
                    if dst == src:
                        continue
                    path = [dst]
                    cur = dst
                    while cur != src:
                        # Balance paths across links
                        if routing == "balanced":
                            n_paths = [n_paths_per_edge_per_class[cls][(pred, cur)] for pred in preds[cur]]
                            pred = preds[cur][n_paths.index(min(n_paths))]
                        # Randomly select shortest paths
                        elif routing == "random":
                            pred = preds[cur][rnd.randint(0, len(preds[cur]) - 1)]
                        # Use the minimum index (what BookSim does)
                        else:
                            pred = preds[cur][0]
                        # n_paths_per_edge_per_class[cls][(pred, cur)] += 1
                        cur = pred
                        path.insert(0, cur)
                    path_src_dst[(src, dst)] = path

    return (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies)

# author: jiaxuming
# time: 2024/2/5/18:47
#############################################################################
N = 16  # 计算矩阵乘法时候的行参数，默认矩阵乘法是N * N 的
now_time = 0  # indicate the time of now
data_id = 0
chiplet_time = []
chiplets_fun_time = 0
chiplet_sim = None


def get_now_time(nodes_by_type):
    global chiplet_time
    min_time = float('inf')
    for chiplet_idx in nodes_by_type['C']:
        min_time = min(min_time, chiplet_time[chiplet_idx])

    return min_time
def trans_data(src_chiplet, end_chiplet, data_id, data_size, router_map, ici_graph):
    global chiplet_sim, now_time, chiplet_time,name_of_design,chiplets_fun_time
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    if data_id % 100 == 0:
        print(data_id)
    while True:
        if (chiplet_time[src_chiplet] <= now_time):
            if dist_src_to_any[src_chiplet][end_chiplet] == float('inf'):
                #输出异常信息告诉用户哪里错了
                print('src_chiplet %d end_chiplet %d can not find the path' % (src_chiplet, end_chiplet))
                file_name = 'warning.txt'
                with open(file_name, 'a') as file:
                    file.write("%d %d %s\n" % (src_chiplet, end_chiplet, 'can not find the path'))
                return
            file_name = './trace_function/bench_%s_%s.txt' % (name_of_design,str(src_chiplet))

            with open(file_name, 'a') as file:
                # 写入字符串到文件
                file.write(
                    "%s %d %d %d %d % s\n" % (str(chiplet_time[src_chiplet]), src_chiplet, end_chiplet, data_size, data_id, 'submit'))
                chiplets_fun_time = max(chiplets_fun_time, chiplet_time[src_chiplet] + node_internal_latency[src_chiplet])
            
            file_name = './trace_function/bench_%s_%s.txt' % (name_of_design,str(end_chiplet))
            with open(file_name, 'a') as file:
                # 写入字符串到文件
                file.write("%s %d %d %d %d %s\n" % (
                    str(max(chiplet_time[src_chiplet] + dist_src_to_any[src_chiplet][end_chiplet],chiplet_time[end_chiplet])),
                    src_chiplet,
                    end_chiplet,
                    data_size,
                    data_id,
                    'receive'
                ))
                chiplets_fun_time = max(chiplets_fun_time, max(chiplet_time[src_chiplet] + dist_src_to_any[src_chiplet][end_chiplet],
                                                             chiplet_time[end_chiplet]))
            chiplet_time[src_chiplet] += phy_latency[src_chiplet]
            chiplet_time[end_chiplet] = max(chiplet_time[src_chiplet] + dist_src_to_any[src_chiplet][end_chiplet],chiplet_time[end_chiplet])

            now_time = get_now_time(nodes_by_type)
            break
        else:
            if now_time == float('inf'):
                break
            else:
                time.sleep(0)
                continue

def get_fun_time(chiplet_time):
    cycle_time = 0
    for i in range(len(chiplet_time)):
        if chiplet_time[i] != float('inf'):
            cycle_time = max(cycle_time, chiplet_time[i])
    return cycle_time     

def mul_sim(chiplet_id, router_map, ici_graph, phy_latency, data_size):
    global now_time, chiplet_time, N, data_id

    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    if chiplet_id == 0:
        # do the things
        for i in range(1, N // 2 + 1, data_size):
            for j in range(1, N + 1, data_size):
                data_id += 1
                trans_data(2, 0, data_id, data_size, router_map, ici_graph)

        for i in range(1, N + 1, data_size):
            for j in range(1, N + 1, data_size):
                data_id += 1
                trans_data(3, 0, data_id, data_size, router_map, ici_graph)

        # 计算上NxN矩阵的上半部分矩阵乘法
        for k in range(1, N // 2 + 1, data_size):
            for i in range(1, N // 2 + 1, data_size):
                for j in range(1, N + 1, data_size):
                    data_id += 1
                    trans_data(0, 3, data_id, data_size, router_map, ici_graph)
        chiplet_time[0] = float('inf')
        now_time = get_now_time(nodes_by_type)
    elif chiplet_id == 1:
        # do the things
        for i in range(N // 2 + 1, N + 1, data_size):
            for j in range(1, N + 1, data_size):
                data_id += 1
                trans_data(3, 1, data_id, data_size, router_map, ici_graph)

        for i in range(1, N + 1, data_size):
            for j in range(1, N + 1, data_size):
                data_id += 1
                trans_data(2, 1, data_id, data_size, router_map, ici_graph)
        for k in range(N // 2 + 1, N + 1, data_size):
            for i in range(N // 2 + 1, N + 1, data_size):
                for j in range(1, N + 1, data_size):
                    data_id += 1
                    trans_data(1, 3, data_id, data_size, router_map, ici_graph)
        chiplet_time[1] = float('inf')
        now_time = get_now_time(nodes_by_type)



def function_sim(router_map, ici_graph, data_size):
    global now_time, chiplet_time, chiplet_sim
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    now_time = 0
    chiplet_time = [0 for i in range(n)]
    chiplet_sim = [chiplet_instance() for i in range(n)]
    t = [Thread(target=mul_sim, args=(chiplet_idx, router_map, ici_graph, phy_latency, data_size)) for chiplet_idx in
         nodes_by_type['C']]
    # 启动线程运行
    for i in range(len(t)):
        t[i].start()
    # 等待所有线程执行完毕
    for i in range(len(t)):
        t[i].join()


def init_chiplet(ici_graph, input_buffer_size, output_buffer_size):
    global chiplet_ins
    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    chiplet_ins = []
    # def __init__(self,chiplet_id = -1,port_num = 0,input_buffer = 0,output_buffer = 0):
    for i in range(n):
        chiplet_ins.append(chiplet_instance(chiplet_id=i, port_num=10, input_buffer=input_buffer_size,
                                            output_buffer=output_buffer_size))


def timing_sim(router_map, ici_graph, file_path,input_buffer_size,output_buffer_size):
    global waiting,priority_map
    (path_src_dst, dist_src_to_any, phy_latency, node_internal_latency, edge_latencies) = router_map
    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    global event_queue
    init_chiplet(ici_graph, input_buffer_size, output_buffer_size)
    event_queue = queue.PriorityQueue()
    with open(file_path, 'r') as fl:
        for line in fl:
            line_data = list(map(int, line.split()))
            next_chiplet = None
            try:
                next_chiplet = path_src_dst[(line_data[1], line_data[2])][1]
            except KeyError:
                next_chiplet = line_data[2]
            port_id = phy_neighbors[(line_data[1], next_chiplet)]
            message_temp = message(data_id=line_data[4], data_src=line_data[1], data_end=line_data[2],
                                   data_type=1, data_size=line_data[3], data_time=line_data[0],
                                   src_port=port_id[0], end_port=port_id[1], next_chiplet=next_chiplet, priority=priority_map[1])
            event_queue.put(message_temp)
    num = 0
    cycle = 0
    while not event_queue.empty():
        num += 1
        # if num == 200:
        #     break
        top_event = event_queue.get()
        cycle = max(cycle,top_event.data_time)
        # if top_event.data_id == 52:
        # print(top_event)
        # if int(top_event.data_id) % 100 == 0:
        # with open('trace_function/result','a') as fl:
        #     fl.write(str(top_event) + '\n')
        if top_event.data_src == top_event.next_chiplet and top_event.data_type == 1:
        #     print(top_event.data_id)
            continue
        if top_event.get_data_type() == 0:
            release_output_buffer(top_event, router_map, phy_neighbors)
        elif top_event.get_data_type() == 1:
            ask_output_buffer(top_event, router_map)
        elif top_event.get_data_type() == 2:
            handle_data_request(top_event, router_map, phy_neighbors)
        elif top_event.get_data_type() == 3:
            creat_event1(top_event, router_map)
        elif top_event.get_data_type() == 4:
            ask_input_room(top_event, router_map, phy_neighbors)
        else:
            print('data type read error')
    print(waiting)
    return cycle


def call_read_data(file_path, file_name, chiplet_num, router_map=None):
    # Construct the command to run the script with arguments
    command = ["python", "Chiplet_Mul/read_data.py", "-fp", file_path, "-fn", file_name, "-cn", str(chiplet_num)]

    # Add router_map argument if it's provided
    if router_map is not None:
        command.extend(["-rm", router_map])

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    if result.returncode == 0:
        print("Script executed successfully!")
        # Print script output (if any)
        print(result.stdout)
    else:
        print("Script execution failed.")
        print(result.stderr)


def plot_simulation_comparison(time_fun, time_timing, cycle_fun, cycle_timing):
    # Create a new figure and set of subplots with adjusted size
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Set the figure title
    fig.suptitle('Simulation Time and Cycles Comparison', fontsize=16)

    # Set the data
    labels = ['Function Simulation', 'Timing Simulation']
    time_data = [round(time_fun, 2), round(time_timing, 2)]
    cycle_data = [cycle_fun, cycle_timing]

    # Set the positions
    x = np.arange(len(labels))  # the label locations

    # Define the width of the bars
    width = 0.4  # the width of the bars

    # Create two bar charts with different colors
    rects1 = ax1.bar(x - width/2, time_data, width, label='Time (seconds)', color='skyblue')
    ax1.set_ylabel('Time (seconds)', fontsize=14)
    ax1.set_ylim(0, max(time_data) * 1.2)  # Set y-axis upper limit to 1.2 times of max time for label space

    # Use the same x-axis but different y-axis
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, cycle_data, width, label='Cycles (count)', color='lightgreen')
    ax2.set_ylabel('Cycles (count)', fontsize=14)
    ax2.set_ylim(0, max(cycle_data) * 1.2)  # Set y-axis upper limit to 1.2 times of max cycles for label space

    # Add data labels
    def autolabel(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1, ax1)
    autolabel(rects2, ax2)

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=14)
    ax1.set_xlabel('Simulation Type', fontsize=14)

    # Set legend position
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=12)

    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space at the top for the title and legend

    # Show the figure
    plt.show()






#计算参数
def compute_metrics(design, routing, data_size,timing,name_of_design,topology_path,input_buffer_size,output_buffer_size):#speed
    global chiplet_time,chiplets_fun_time
    # Technology node

    #need:205854
    technology = hlp.read_file(filename=design["technology_nodes_file"])
    # Chiplets
    chiplets = hlp.read_file(filename=design["chiplets_file"])
    # Placement
    #need
    placement = hlp.read_file(filename=design["chiplet_placement_file"])
    # Topology
    topology = hlp.read_file(filename=design["ici_topology_file"])
    # Packaging
    packaging = hlp.read_file(filename=design["packaging_file"])
    # Validate design
    if not vld.validate_design(design, technology, chiplets, placement, topology, packaging):
        print("warning: This design contains validation errors - the RapidChiplet toolchain might fail.")
    # Only construct the ICI graph if we need it (i.e. if latency or throughput are computed)
    ici_graph = construct_ici_graph(chiplets, placement, topology)
    if not vld.validate_ici_graph(ici_graph):
        print("warning: The ICI topology contains validation errors - the RapidChiplet toolchain might fail.")
    ici_routing = construct_ici_routing(ici_graph, routing)
    area_summary = compute_area_summary(chiplets, placement)
    (link_summary, link_lengths_internal) = compute_link_summary(chiplets, placement, topology, packaging)
    manufacturing_cost = compute_manufacturing_cost(technology, chiplets, placement, packaging, area_summary)
    # 测试
    (c, r, n, neighbors, relay_map, nodes_by_type, phy_neighbors) = ici_graph
    router_map = compute_router_map(technology, chiplets, placement, packaging, ici_graph, ici_routing,link_lengths_internal, ici_routing)
    if data_size == None:
        data_size = 1
    data_size = int(data_size)
    time_fun,time_timing= None,None
    cycle_timing = None
    
    time_fun = time.time()
    time_timing = time.time()
    function_sim(router_map, ici_graph, data_size)
    cycle_fun = chiplets_fun_time
    time_fun = time.time() - time_fun
    trace_name = 'bench_' + name_of_design

     
    if timing:
        print("Timing simulation is enabled.")
        call_read_data("trace_function", trace_name, n, topology_path)
        #生成仿真文件路径./trace_function/bench_mesh_1x2_s2e.txt
        file_trace = './trace_function/bench_%s_s2e.txt' % name_of_design
        
        cycle_timing = timing_sim(router_map, ici_graph, file_trace,input_buffer_size,output_buffer_size)
        time_timing = time.time() - time_timing
        #检查输入输出缓冲区文件是否存在
        for i in range(n):
            for x in range(chiplet_ins[i].port_num):
                if len(chiplet_ins[i].input_data[x]):
                    print('input buffer error')
                if len(chiplet_ins[i].output_data[x]):
                    print('output buffer error')
                    
    # 生成一个条形图，对比仿真时间和cycle时间
    print("Function Simulation Time: %.2f seconds" % time_fun)
    print("Function Simulation Cycles: %d" % cycle_fun)
    print("Timing Simulation Time: %.2f seconds" % time_timing)
    print("Timing Simulation Cycles: %d" % cycle_timing)
    plot_simulation_comparison(time_fun, time_timing, cycle_fun, cycle_timing)
    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #需要的设计文件
    parser.add_argument("-df", "--design_file", required=True, help="Path to the \"design\" input file")
    #路由方式
    parser.add_argument("-r", "--routing", required=False,
                        help="Use the non-default \"balanced\" or \"random\" routing")
    #数据大小，默认为1
    parser.add_argument("-ds", "--data_size", required=False, help='Get the data size')
    #是否进行时序仿真
    parser.add_argument("-t", "--timing", action="store_true", help='Whether to perform timing simulation')
    #加入拓扑文件路径
    parser.add_argument("-tp", "--topology_path", required=False, help='Path to the topology path file')
    #输入缓冲区大小
    parser.add_argument("-ib", "--input_buffer", required=False, help='Input buffer size')
    #输出缓冲区大小
    parser.add_argument("-ob", "--output_buffer", required=False, help='Output buffer size')
    args = parser.parse_args()
    #获取路径文件名
    name_of_design = args.design_file.split('/')[-1].split('.')[0]
    #如果输入缓冲区没设置，默认为1
    if args.input_buffer == None:
        args.input_buffer = 1
    #如果输出缓冲区没设置，默认
    if args.output_buffer == None:
        args.output_buffer = 1
    # print(file_name)
    design = hlp.read_file(filename=args.design_file)
    compute_metrics(design=design,
                    routing=args.routing,
                    data_size=args.data_size,
                    timing=args.timing,
                    name_of_design=name_of_design,
                    topology_path=args.topology_path,
                    input_buffer_size=int(args.input_buffer),
                    output_buffer_size=int(args.output_buffer))
#  python Multi-chiplet.py --design_file  inputs/designs/mesh_1x2.json   --timing --topology_path Chiplet_Mul/routing_map -ib 1 -ob 1
