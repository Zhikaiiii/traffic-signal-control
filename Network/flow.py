"""The module for generating .xml format files of traffic flow definition,
refer to Aven's blog(http://www.itstech.club/)."""
from __future__ import division
from __future__ import unicode_literals
import os
import numpy as np

from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree


class XMLHelper(object):
    """The class for manipulate the simple .xml file."""
    @staticmethod
    def read_xml(xml_file):
        """Read xml file."""
        tree = ElementTree()
        tree.parse(xml_file)
        return tree

    @staticmethod
    def write_xml(tree, output_file):
        """Write xml file."""
        tree.write(output_file, encoding="utf-8", xml_declaration=False)

    @staticmethod
    def create_node(tag, properties):
        """Create a node."""
        return Element(tag, properties)

    @staticmethod
    def add_child_node(parent_node, element):
        """Add a child node."""
        parent_node.append(element)

    @staticmethod
    def del_node(parent_node, tag):
        """Delete the specific node."""
        for child_node in reversed(parent_node.getchildren()):
            if child_node.tag == tag:
                parent_node.remove(child_node)

    @staticmethod
    def get_nodes(parent_node, tag):
        """Get the corresponding node."""
        nodes = []
        for child_node in parent_node.getchildren():
            if child_node.tag == tag:
                nodes.append(child_node)
        return nodes

    @staticmethod
    def add_linesep(parent_node):
        """Add indent."""
        for child_node in parent_node.getchildren():
            if not child_node.tail or not child_node.tail.strip():
                child_node.tail = os.linesep


class FlowHelper(object):
    """The class supporting basic operation for generate .rou.xml file."""
    def __init__(self, flow_file):
        """Bind to a .rou.xml."""
        if os.path.exists(flow_file):
            self.tree = XMLHelper.read_xml(flow_file)
        else:
            print("The .rou.xml file doesn't exist.")

    def add_flow_by_dict(self, flow_dict_list):
        """Add <flow/> with attributes in the dictionary."""
        for flow_dict in flow_dict_list:
            element = XMLHelper.create_node("flow", flow_dict)
            XMLHelper.add_child_node(self.tree.getroot(), element)

    def remove_all(self):
        """Remove all of the flow definition."""
        XMLHelper.del_node(self.tree.getroot(), "flow")
        XMLHelper.del_node(self.tree.getroot(), "vehicle")

    def write_xml(self, output_file):
        """Keep the begin time increasing, and write to the .rou.xml."""
        nodes_list = XMLHelper.get_nodes(self.tree.getroot(), "flow")
        nodes_list.sort(key=lambda n: int(n.get('begin')))
        XMLHelper.del_node(self.tree.getroot(), "flow")
        for node in nodes_list:
            XMLHelper.add_child_node(self.tree.getroot(), node)
        XMLHelper.add_linesep(self.tree.getroot())
        XMLHelper.write_xml(self.tree, output_file)

    def add_flow(self, flow_id, vehicle_type, from_edge, to_edge, begin, end, vehsPerHour, depart_lane="random"):
        """Add flow by attributes."""
        flow_dict = {'id': flow_id, 'type': vehicle_type,
                     'from': from_edge, 'to': to_edge,
                     'begin': str(int(begin)), 'end': str(int(end)),
                     'vehsPerHour': str(int(vehsPerHour)), 'departLane': depart_lane,
                     'departSpeed': 'random'}
        node = XMLHelper.create_node("flow", flow_dict)
        XMLHelper.add_child_node(self.tree.getroot(), node)
    # 在不同时刻产生不同密度的交通流
    # 正弦变化
    def add_sin_flow(self, id_prefix, vehicle_type, from_edge, to_edge, begin, end, n_slots,
                     min_volume, max_volume, offset):
        """Add sin(t) time-vary flow."""
        time_slots = np.linspace(begin, end, n_slots + 1)
        angles = np.linspace(0+offset, 360+offset, n_slots)
        volumes = (max_volume + min_volume)/2 + (max_volume - min_volume)/2 * np.sin(angles * np.pi / 180)
        for i, vol in enumerate(volumes):
            flow_dict = {'id': id_prefix + str(i), 'type': vehicle_type,
                         'from': from_edge, 'to': to_edge,
                         'begin': str(int(time_slots[i])), 'end': str(int(time_slots[i+1])),
                         'vehsPerHour': str(int(vol)), 'departLane': "random",
                         'departSpeed': 'random'}
            node = XMLHelper.create_node("flow", flow_dict)
            XMLHelper.add_child_node(self.tree.getroot(), node)

    def add_linear_flow(self, id_prefix, vehicle_type, from_edge, to_edge, begin, end, n_slots,
                        begin_volume, end_volume):
        """Add linear flow"""
        time_slots = np.linspace(begin, end, n_slots + 1)
        add_volume = (end_volume - begin_volume) / (2 * n_slots)
        for i in range(n_slots):
            vol = begin_volume + (2 * i + 1) * add_volume
            flow_dict = {'id': id_prefix + str(i), 'type': vehicle_type,
                         'from': from_edge, 'to': to_edge,
                         'begin': str(int(time_slots[i])), 'end': str(int(time_slots[i+1])),
                         'vehsPerHour': str(int(vol)), 'departLane': 'random',
                         'departSpeed': 'random'}
            node = XMLHelper.create_node("flow", flow_dict)
            XMLHelper.add_child_node(self.tree.getroot(), node)
