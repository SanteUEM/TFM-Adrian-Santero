from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'pedcom_actuation'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'audio'), glob('audio/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adrian Santero Alonso',
    maintainer_email='adriansantero99@gmail.com',
    description='Actuacion multisensorial: iluminacion, audio y movimiento.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'led_driver_node = pedcom_actuation.led_driver_node:main',
            'audio_player_node = pedcom_actuation.audio_player_node:main',
            'head_motion_node = pedcom_actuation.head_motion_node:main',
        ],
    },
)
