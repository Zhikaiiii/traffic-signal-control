import os
import random
from Network.flow import FlowHelper

PATH = '../Data/networks/data/'
INTERSECTION_DISTANCE = 500
END_DISTANCE = 200
N_INTERSECTION = 2
SPEED_LIMIT = 16.67

# 产生路网结构
class NetworkGenerator():
    def __init__(self, name_network):
        self.name_network = name_network
        self.path = PATH
        self.i_distance = INTERSECTION_DISTANCE
        self.e_distance = END_DISTANCE
        self.n_intersection = N_INTERSECTION
    
    def create_network(self, init_density, seed=None, thread=None):
        print('generating nod file...')
        self.gen_nod_file()
        print('generating typ file...')
        self.gen_typ_file()
        print('generating edg file...')
        self.gen_edg_file()
        print('generating con file...')
        self.gen_con_file()
        print('generating tll file...')
        self.gen_tll_file()
        print('generating net file...')
        self.gen_net_file()
        print('generating rou file...')
        self.gen_rou_file(init_density, seed)
        print('generating add file...')
        self.gen_add_file()
        print('generating sim file...')
        self.gen_sumocfg(thread)

    def _write_file(self, path, content):
        with open(path, 'w') as f:
            f.write(content)

    # 产生节点: 位置
    def gen_nod_file(self):
        path = self.path + self.name_network +'.nod.xml'
        node_context = '<nodes>\n'
        node_str = '  <node id="%s" x="%.2f" y="%.2f" type="%s"/>\n'
        length = self.i_distance*self.n_intersection
        index = 0
        for y in range(0, length, self.i_distance):
            for x in range(0, length, self.i_distance):
                node_context += node_str % ('I' + str(index), x, y, 'traffic_light')
                index += 1
        index = 0
        for x in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), x, -self.e_distance, 'priority')
            index += 1
        for x in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), x, length-self.i_distance+self.e_distance, 'priority')
            index += 1
        for y in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), -self.e_distance, y, 'priority')
            index += 1
        for y in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), length-self.i_distance+self.e_distance, y, 'priority')
            index += 1
        node_context += '</nodes>\n'
        self._write_file(path, node_context)

    # 道路类型信息：车道，速度限制
    def gen_typ_file(self):
        path = self.path + self.name_network +'.typ.xml'
        type_context = '<types>\n'
        type_context += '  <type id="a" numLanes="3" speed="%.2f"/>\n' % SPEED_LIMIT
        type_context += '</types>\n'
        self._write_file(path, type_context)
    
    def _gen_edg_str(self, edge_str, from_node, to_node, edge_type):
        edge_id = '%s_%s' %(from_node, to_node)
        return edge_str %(edge_id, from_node, to_node, edge_type)

    # 道路具体设置： 连接的两个junction
    def gen_edg_file(self):
        path = self.path + self.name_network +'.edg.xml'
        edges_context = '<edges>\n'
        edges_str = '  <edge id="%s" from="%s" to="%s" type="%s"/>\n'
        # node_pair = [('P0','I0'),('P1','I1'),('P2','I2'),
        #             ('P3','I6'),('P4','I7'),('P5','I8'),
        #             ('P6','I0'),('P7','I3'),('P8','I6'),
        #             ('P9','I2'),('P10','I5'),('P11','I8'),
        #             ('I0','I1'),('I1','I2'),('I3','I4'),('I4','I5'),('I6','I7'),('I7','I8'),
        #             ('I0','I3'),('I1','I4'),('I2','I5'),
        #             ('I3','I6'),('I4','I7'),('I5','I8')]
        node_pair = [('P0','I0'),('P1','I1'),
                    ('P2','I2'),('P3','I3'),
                    ('P4','I0'),('P5','I2'),
                    ('P6','I1'),('P7','I3'),
                    ('I0','I1'),('I2','I3'),
                    ('I0','I2'),('I1','I3')]
        for (i1,i2) in node_pair:
            edges_context += self._gen_edg_str(edges_str, i1, i2, 'a')
            edges_context += self._gen_edg_str(edges_str, i2, i1, 'a')
        edges_context += '</edges>\n'
        self._write_file(path, edges_context)
    
    def get_con_str(self, con, from_node, cur_node, to_node, from_lane, to_lane):
        from_edge = '%s_%s' % (from_node, cur_node)
        to_edge = '%s_%s' % (cur_node, to_node)
        return con % (from_edge, to_edge, from_lane, to_lane)
    # 定义交通轨迹
    def _gen_con_node(self, con_str, cur_node, n_node, s_node, w_node, e_node):
        str_cons = ''
        # go-through
        str_cons += self.get_con_str(con_str, s_node, cur_node, n_node, 0, 0)
        str_cons += self.get_con_str(con_str, n_node, cur_node, s_node, 0, 0)
        str_cons += self.get_con_str(con_str, s_node, cur_node, n_node, 1, 1)
        str_cons += self.get_con_str(con_str, n_node, cur_node, s_node, 1, 1)
        # str_cons += self._gen_con_str(con_str, s_node, cur_node, n_node, 1, 2)
        # str_cons += self._gen_con_str(con_str, n_node, cur_node, s_node, 1, 2)
        str_cons += self.get_con_str(con_str, w_node, cur_node, e_node, 0, 0)
        str_cons += self.get_con_str(con_str, e_node, cur_node, w_node, 0, 0)
        str_cons += self.get_con_str(con_str, w_node, cur_node, e_node, 1, 1)
        str_cons += self.get_con_str(con_str, e_node, cur_node, w_node, 1, 1)
        # str_cons += self._gen_con_str(con_str, w_node, cur_node, e_node, 1, 2)
        # str_cons += self._gen_con_str(con_str, e_node, cur_node, w_node, 1, 2)
        # left-turn
        str_cons += self.get_con_str(con_str, s_node, cur_node, w_node, 2, 2)
        str_cons += self.get_con_str(con_str, n_node, cur_node, e_node, 2, 2)
        str_cons += self.get_con_str(con_str, w_node, cur_node, n_node, 2, 2)
        str_cons += self.get_con_str(con_str, e_node, cur_node, s_node, 2, 2)
        # str_cons += self._gen_con_str(con_str, s_node, cur_node, w_node, 2, 1)
        # str_cons += self._gen_con_str(con_str, n_node, cur_node, e_node, 2, 1)
        # str_cons += self._gen_con_str(con_str, w_node, cur_node, n_node, 2, 1)
        # str_cons += self._gen_con_str(con_str, e_node, cur_node, s_node, 2, 1)
        # right-turn
        str_cons += self.get_con_str(con_str, s_node, cur_node, e_node, 0, 0)
        str_cons += self.get_con_str(con_str, n_node, cur_node, w_node, 0, 0)
        str_cons += self.get_con_str(con_str, w_node, cur_node, s_node, 0, 0)
        str_cons += self.get_con_str(con_str, e_node, cur_node, n_node, 0, 0)
        return str_cons
    
    def gen_con_file(self):
        path = self.path + self.name_network +'.con.xml'
        connections_context = '<connections>\n'
        connections_str = '  <connection from="%s" to="%s" fromLane="%d" toLane="%d"/>\n'
        # node_pair = [('I0','I3','P0','P6','I1'),('I1','I4','P1','I0','I2'),('I2','I5','P2','I1','P9'),
        #             ('I3','I6','I0','P7','I4'),('I4','I7','I1','I3','I5'),('I5','I8','I2','I4','P10'),
        #             ('I6','P3','I3','P8','I7'),('I7','P4','I4','I6','I8'),('I8','P5','I5','I7','P11')]
        node_pair = [('I0','I2','P0','P4','I1'),('I1','I3','P1','I0','P6'),
                    ('I2','P2','I0','P5','I3'),('I3','P3','I1','I2','P7')]
        for (cur,n,s,w,e) in node_pair:
            connections_context += self._gen_con_node(connections_str, cur, n, s, w, e)
        connections_context += '</connections>\n'
        self._write_file(path, connections_context)

    # 定义交通信号灯
    def gen_tll_file(self):
        random.seed()
        path = self.path + self.name_network +'.tll.xml'
        tls_str = '  <tlLogic id="%s" programID="0" offset="%d" type="actuated">\n'
        phase_str = '    <phase duration="%d" minDur="%d" maxDur="%d" state="%s"/>\n'
        tls_context = '<additional>\n'
        phases = [('GGGrrrrrGGGrrrrr', 15, 5, 30), ('yyyrrrrryyyrrrrr',2, 2, 2),
                 ('rrrGrrrrrrrGrrrr', 15, 5, 30), ('rrryrrrrrrryrrrr',2, 2, 2),
                 ('rrrrGGGrrrrrGGGr', 15, 5, 30), ('rrrryyyrrrrryyyr',2, 2, 2),
                 ('rrrrrrrGrrrrrrrG', 15, 5, 30), ('rrrrrrryrrrrrrry',2, 2, 2)]
        for ind in range(self.n_intersection*self.n_intersection):
            offset = random.randint(0, 20)
            node_id = 'I' + str(ind)
            tls_context += tls_str % (node_id, offset)
            for (state, duration, min_dur, max_dur) in phases:
                tls_context += phase_str % (duration, min_dur, max_dur, state)
            tls_context += '  </tlLogic>\n'
        tls_context += '</additional>\n'
        self._write_file(path, tls_context)
    
    def gen_net_file(self):
        config_context = '<configuration>\n  <input>\n'
        config_context += '    <edge-files value="'+self.name_network+'.edg.xml"/>\n'
        config_context += '    <node-files value="'+self.name_network+'.nod.xml"/>\n'
        config_context += '    <type-files value="'+self.name_network+'.typ.xml"/>\n'
        config_context += '    <tllogic-files value="'+self.name_network+'.tll.xml"/>\n'
        config_context += '    <connection-files value="'+self.name_network+'.con.xml"/>\n'
        config_context += '  </input>\n  <output>\n'
        config_context += '    <output-file value="'+self.name_network+'.net.xml"/>\n'
        config_context += '  </output>\n</configuration>\n'
        path = self.path + self.name_network +'.netccfg'
        self._write_file(path, config_context)
        os.system('netconvert -c '+ path)

    # 路径定义
    def gen_rou_file(self, init_density, seed=None):
        if seed is not None:
            random.seed(seed)
        path = self.path + self.name_network +'.rou.xml'
    #   ext_flow = '  <flow id="f:%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
        flows_context = '<routes>\n'
        flows_context += '  <vType id="type1" length="5" accel="2.6" decel="4.5"/>\n'
        flows_context += self._init_flow(init_density)
        flows_context += '</routes>\n'
        self._write_file(path, flows_context)
        flows = FlowHelper(path)
        flows.add_sin_flow('flow_sin_1', 'type1', 'P0_I0', 'I7_P4', 0, 3600, 20, 100, 600, 0)
        flows.add_sin_flow('flow_sin_2', 'type1', 'P7_I3', 'I2_P2', 0, 3600, 20, 80, 500, 90)
        flows.add_sin_flow('flow_sin_3', 'type1', 'P9_I2', 'I6_P3', 0, 3600, 20, 100, 700, 180)
        # flows.add_sin_flow('flow_sin_4', 'type1', 'P4_I7', 'I0_P6', 0, 3600, 20, 120, 600, 270)
        flows.add_linear_flow('flow_line_1', 'type1', 'P5_I8', 'I1_P1', 0, 3600, 20, 0, 400)
        flows.add_linear_flow('flow_line_2', 'type1', 'P10_I5', 'I6_P8', 0, 3600, 20, 400, 0)
        flows.add_linear_flow('flow_line_3', 'type1', 'P6_I0', 'I8_P11', 0, 3600, 20, 200, 500)
        # flows.add_sin_flow('flow_sin_1', 'type1', 'P0_I0', 'I7_P4', 0, 3600, 20, 100, 600, 0)
        # flows.add_sin_flow('flow_sin_2', 'type1', 'P7_I3', 'I2_P2', 0, 3600, 20, 80, 500, 45)
        # flows.add_sin_flow('flow_sin_3', 'type1', 'P9_I2', 'I6_P3', 0, 3600, 20, 100, 700, 90)
        # flows.add_linear_flow('flow_line_1', 'type1', 'P5_I8', 'I1_P1', 0, 3600, 20, 0, 400)
        # flows.add_linear_flow('flow_line_2', 'type1', 'P10_I5', 'I6_P8', 0, 3600, 20, 400, 0)
        # flows.add_linear_flow('flow_line_3', 'type1', 'P6_I0', 'I8_P11', 0, 3600, 20, 200, 500)
        flows.write_xml(path)
    
    def _init_flow(self, init_density):
        init_flow = '  <flow id="init:%s" departPos="random_free" from="%s" to="%s" begin="0" end="1" departLane="random" departSpeed="random" number="%d" type="type1"/>\n'
        init_flow_context = ''
        car_num = int(30 * init_density)
        # destination = ['I0_P0', 'I0_P6', 'I1_P1', 'I2_P2', 'I2_P9', 'I5_P10',
        #               'I8_P11', 'I8_P5', 'I7_P4', 'I6_P3', 'I6_P8', 'I3_P7']
        destination = ['I0_P0', 'I0_P4', 'I1_P1', 'I1_P6',
                      'I2_P2', 'I2_P5', 'I3_P3', 'I3_P7']
        def get_od(node1, node2, k):
            ori_edge = '%s_%s' % (node1, node2)
            dest = random.choice(destination)
            return init_flow % (str(k), ori_edge, dest, car_num)
        k = 1
        for i in range(0, 3, 2):
            node1 = 'I' + str(i)
            node2 = 'I' + str(i + 1)
            init_flow_context += get_od(node1, node2, k)
            k += 1
            init_flow_context += get_od(node2, node1, k)
            k += 1
        # for i in range(0, 8, 3):
        #     for j in range(2):
        #         node1 = 'I' + str(i + j)
        #         node2 = 'I' + str(i + j + 1)
        #         init_flow_context += get_od(node1, node2, k)
        #         k += 1
        #         init_flow_context += get_od(node2, node1, k)
        #         k += 1
        for i in range(0, 2):
            # for j in range(0, 6, 3):
            node1 = 'I' + str(i)
            node2 = 'I' + str(i + 2)
            init_flow_context += get_od(node1, node2, k)
            k += 1
            init_flow_context += get_od(node2, node1, k)
            k += 1
        # avenues
        # for i in range(0, 3):
        #     for j in range(0, 6, 3):
        #         node1 = 'I' + str(i + j)
        #         node2 = 'I' + str(i + j + 3)
        #         init_flow_context += get_od(node1, node2, k)
        #         k += 1
        #         init_flow_context += get_od(node2, node1, k)
        #         k += 1
        return init_flow_context
    
    def _gen_add_str(self, ild_str, from_node, to_node, n_lane):
        edge_id = '%s_%s' % (from_node, to_node)
        edge_add_str = ''
        for i in range(n_lane):
            edge_add_str += ild_str % (edge_id, i, edge_id, i)
        return edge_add_str
    
    def _gen_add_node(self, ild_str, cur_node, n_node, s_node, w_node, e_node):
        node_add_str = ''
        node_add_str += self._gen_add_str(ild_str, n_node, cur_node, n_lane=3)
        node_add_str += self._gen_add_str(ild_str, s_node, cur_node, n_lane=3)
        node_add_str += self._gen_add_str(ild_str, w_node, cur_node, n_lane=3)
        node_add_str += self._gen_add_str(ild_str, e_node, cur_node, n_lane=3)
        return node_add_str

    def gen_add_file(self):
        path = self.path + self.name_network +'.add.xml'
        ild_context = '<additional>\n'
        ild_str = '  <laneAreaDetector file="ild.out" freq="1" id="%s_%d" lane="%s_%d"\
             pos="-100" endPos="-1"/>\n'
        # node_pair = [('I0','I3','P0','P6','I1'),('I1','I4','P1','I0','I2'),('I2','I5','P2','I1','P9'),
        #             ('I3','I6','I0','P7','I4'),('I4','I7','I1','I3','I5'),('I5','I8','I2','I4','P10'),
        #             ('I6','P3','I3','P8','I7'),('I7','P4','I4','I6','I8'),('I8','P5','I5','I7','P11')]
        node_pair = [('I0','I2','P0','P4','I1'),('I1','I3','P1','I0','P6'),
                    ('I2','P2','I0','P5','I3'),('I3','P3','I1','I2','P7')]
        for (cur,n,s,w,e) in node_pair:
            ild_context += self._gen_add_node(ild_str, cur, n, s, w, e)
        ild_context += '</additional>\n'
        self._write_file(path, ild_context)
    
    def gen_sumocfg(self, thread=None):
        path = self.path + self.name_network +'.sumocfg'
        if thread is None:
            out_file = self.name_network+'.rou.xml'
        else:
            out_file = self.name_network+'_%d.rou.xml' % int(thread)
        config_context = '<configuration>\n  <input>\n'
        config_context += '    <net-file value="'+self.name_network+'.net.xml"/>\n'
        config_context += '    <route-files value="%s"/>\n' % out_file
        config_context += '    <additional-files value="'+self.name_network+'.add.xml"/>\n'
        config_context += '  </input>\n  <time>\n'
        config_context += '    <begin value="0"/>\n    <end value="3600"/>\n'
        config_context += '  </time>\n</configuration>\n'
        self._write_file(path, config_context)

if __name__=='__main__':
    ng = NetworkGenerator('Grid4_act')
    ng.create_network(init_density=0.2, seed=49)
    # ng.gen_net_file()
