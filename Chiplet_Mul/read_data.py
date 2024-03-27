#读取路径文件，并且将多芯片文件合成一个以时间为顺序的多芯片��结文件
import argparse
from pathlib import Path
file_path = '../trace_function/'
file_name = ''
chiplet_num = -1
def get_route_map(path_router_map):
    x = 0
    position_map = {}
    with open(path_router_map,'r') as fl:
        for line in fl:
            row = list(map(int,line.split('-')))
            for i in range(len(row)):
                position_map[row[i]] = (x,i)
        x += 1

    return position_map


class trace_std():

    def __init__(self,cycle = -1,src_chiplet = -1,end_chiplet = -1,data_size = 0,data_id = -1,data_type = None):
        self.cycle = int(cycle)
        self.src_chiplet = int(src_chiplet)
        self.end_chiplet = int(end_chiplet)
        self.data_size = int(data_size)
        self.data_id = int(data_id)
        self.data_type = data_type

    def __lt__(self, other):
        return self.cycle < other.cycle

    def __repr__(self):
        return f"{self.cycle} {self.src_chiplet} {self.end_chiplet} {self.data_id} {self.data_size}"

    def save(self,router_map):
        global file_path,file_name
        file = file_path + '/' + file_name + '.txt'
        if router_map == None:
            with open(file,'a') as fl:
                fl.write('%d %d %d %d %d %d\n'%(self.cycle,0,self.src_chiplet,0,self.end_chiplet,self.data_size))
            return
        else:
            with open(file,'a') as fl:
                fl.write('%d %d %d %d %d %d\n'%(self.cycle,
                                                router_map[self.src_chiplet][0],router_map[self.src_chiplet][1],
                                                router_map[self.end_chiplet][0],router_map[self.end_chiplet][1],
                                                self.data_size))
        file = file_path + '/' + file_name + '_s2e.txt'
        with open(file, 'a') as fl:
            fl.write('%d %d %d %d %d\n' % (self.cycle, self.src_chiplet, self.end_chiplet, self.data_size,self.data_id))



def save_all(file_name,chiplet_num,router_map):
    ins = 0#指令个数
    chiplets_trace = []
    for i in range(chiplet_num):
        open_file_name = file_path + '/' + file_name + '_' + str(i) + '.txt'
        if not Path(open_file_name).exists():
            continue
        with open(open_file_name,'r') as fl:
            for line in fl:
                if line.split()[-1][0] == 's':
                    chiplets_trace.append(trace_std(*line.split()))


    chiplets_trace = sorted(chiplets_trace)

    for x in chiplets_trace:
        x.save(router_map)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", "--file_path", required=True, help="Path to the  input file")
    parser.add_argument("-rm","--router_map",required=False,help="path to the router_map")
    parser.add_argument("-fn", "--file_name", required=True, help="file prefix name of the input file ")
    parser.add_argument("-cn", "--chiplet_num", required=True, help="the chiplet number")
    args = parser.parse_args()
    file_path = args.file_path
    path_router_map = args.router_map

    file_name = args.file_name
    if path_router_map == None:
        router_map = None
    else:
        router_map = get_route_map(path_router_map)
    chiplet_num = int(args.chiplet_num)
    save_all(file_name,chiplet_num,router_map)



# python Chiplet_Mul/read_data.py -fp  trace_function -fn bench -cn 4 -rm Chiplet_Mul/routing_map



