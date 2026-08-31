from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'pedcom_behaviour'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adrian Santero Alonso',
    maintainer_email='adriansantero99@gmail.com',
    description='Fusion temporal, gestor de comportamiento y gobernanza de datos.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'affect_fusion_node = pedcom_behaviour.affect_fusion_node:main',
            'behaviour_manager_node = '
            'pedcom_behaviour.behaviour_manager_node:main',
            'privacy_guard_node = pedcom_behaviour.privacy_guard_node:main',
            'session_logger_node = pedcom_behaviour.session_logger_node:main',
        ],
    },
)
