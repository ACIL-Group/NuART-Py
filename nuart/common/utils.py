"""
   Copyright 2018 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__author__ = 'Islam Elnabarawy'


def scale_range(x, x_range, y_range=(0.0, 1.0)):
    """
    scale the number x from the range specified by x_range to the range specified by y_range

    :param x: the number to scale
    :type x: float
    :param x_range: the number range that x belongs to
    :type x_range: tuple
    :param y_range: the number range to convert x to, defaults to (0.0, 1.0)
    :type y_range: tuple
    :return: the scaled value
    :rtype: float
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    return (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min


def normalize(p, data_ranges):
    for i in range(len(p)):
        p[i] = scale_range(p[i], data_ranges[i])


def denormalize(p, data_ranges):
    for i in range(len(p)):
        p[i] = scale_range(p[i], x_range=(0.0, 1.0), y_range=data_ranges[i])
