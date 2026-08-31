from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'pedcom_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adrian Santero Alonso',
    maintainer_email='adriansantero99@gmail.com',
    description='Subsistema de percepcion facial y afectiva de Pediatric-Companion.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = pedcom_perception.perception_node:main',
            'sequence_injector_node = '
            'pedcom_perception.sequence_injector_node:main',
        ],
    },
)
