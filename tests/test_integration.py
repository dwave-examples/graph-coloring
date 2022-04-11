# Copyright 2022 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import unittest
import os
import sys

import graph_coloring
import map_coloring

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class IntegrationTests(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_graph_coloring(self):
        """run graph_coloring.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'graph_coloring.py')
        subprocess.check_output([sys.executable, demo_file])

    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_build_graph_cqm(self):
        """Test that CQM built has correct number of variables"""

        num_colors = 7
        num_nodes = 50

        G, _ = graph_coloring.build_graph(num_nodes)

        cqm = graph_coloring.build_cqm(G, num_colors)

        self.assertEqual(len(cqm.variables), num_colors*num_nodes)

    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_map_coloring(self):
        """run map_coloring.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'map_coloring.py')
        subprocess.check_output([sys.executable, demo_file])

    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_canada_map(self):
        """Test that Canada map option works"""

        input_shp_file = os.path.join(project_dir, 'shp_files/canada/canada.shp')

        _, state_neighbors = map_coloring.get_state_info(input_shp_file)

        G = map_coloring.build_graph(state_neighbors)

        cqm = map_coloring.build_cqm(G, 5)

        sample = map_coloring.run_hybrid_solver(cqm)

        colors_used = max(sample.values())+1

        self.assertLessEqual(colors_used, 5)

if __name__ == '__main__':
    unittest.main()
