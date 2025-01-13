import argparse

class _Generation_ArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(_Generation_ArgParser, self).__init__( **kwargs)
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        generator_args = self.add_argument_group('generator')
        generator_args.add_argument(
            '-g','--generator_config',type=str,default='config/generastor.yaml',help='generator config file'
        )

        overwrite_args = self.add_argument_group('overwrite generator config')
        overwrite_args.add_argument('--model_path', type=str, help='model path')
        overwrite_args.add_argument('--library_path', type=str, help='library path')
        overwrite_args.add_argument('--library_builtin_model_path', type=str, help='builtin model path')

        opt_args = self.add_argument_group('optional')
        opt_args.add_argument('-o', '--output_path', type=str, help='output file name')
        opt_args.add_argument('-n', '--num_samples', type=int, help='number of generation', default=1)
        opt_args.add_argument('--seed', type=int, help='explicit random seed')
        opt_args.add_argument('-q', action='store_true', help='no print sampling script message')

class Scaffold_Generation_ArgParser(_Generation_ArgParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        scaffold_args = self.add_argument_group('scaffold-based generation')
        scaffold_args.add_argument('-s', '--scaffold', type=str, help='scaffold smi')
        scaffold_args.add_argument('-S', '--scaffold_path', type=str, help='scaffold smi path')

