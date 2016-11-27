from setuptools import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
install_reqs_parsed = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

setup(
    name='bb_plotter',
    description='Plotting helper for beesbook',
    version='0.11',
    author='Kadir Tugan',
    author_email='ktugan@users.noreply.github.com',
    url='https://github.com/ktugan/bb_plotter/',
    install_requires=install_reqs_parsed,
    dependency_links=dep_links,
    packages=['bb_plotter'],
)