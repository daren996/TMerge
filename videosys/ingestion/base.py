import threading

class Collector:
    def __init__(self, operators=None):
        self.operators = operators if operators is not None else []

    def emit(self, tables=None):
        if self.operators is None or len(self.operators) == 0:
            return
        for operator in self.operators:
            operator.process(tables)

class Operator:
    def __init__(self, config=None):
        '''
        Params:
        config: configuration for this operator.
        '''
        self.config = {} if config is None else config
        self.collector = None
        self.context = None
    
    def _get_config(self, key, default_value=None):
        if key in self.config:
            return self.config[key]
        if default_value is None:
            raise LookupError('config "{}" is required'.format(key))
        return default_value

    def setup(self, context=None, collector=None, controller=None):
        '''
        context: the context of this operator
        collector: the collector for this operator.
        '''
        if context is not None:
            self.context = context
        if collector is not None:
            self.collector = collector
        if controller is not None:
            self.controller = controller

    def prepare(self):
        pass

    def process(self, tables):
        pass

    def cleanup(self):
        pass

class Source(Operator):
    def has_next(self):
        pass

class Context:
    def __init__(self):
        self.dict = {}

    def get(self, key):
        return self.dict[key]

    def put(self, key, value):
        self.dict[key] = value

    def has(self, key):
        return key in self.dict

class PipelineController:
    def __init__(self):
        self.__cv = threading.Condition()
        self.end = False

    # def pause(self):
    #     with self.__cv:
    #         self.__cv.wait()

    # def resume(self):
    #     with self.__cv:
    #         self.__cv.notify()

    def stop(self):
        self.end = True

class Pipeline:
    def __init__(self, start_op, all_ops, controller=None):
        self.start_op = start_op
        self.all_ops = all_ops
        self.controller = controller if controller is not None else PipelineController()
        if controller is not None:
            for operator in all_ops:
                operator.setup(controller=controller)

    def start(self):
        # prepare.
        for operator in self.all_ops:
            operator.prepare()

        while self.start_op.has_next() and not self.controller.end:
            self.start_op.process()

        for operator in self.all_ops:
            operator.cleanup()


class AbstractPipelineBuilder:
    '''
    Build pipeline
    '''
    def add_operator(self, operator):
        pass

    def build(self):
        pass

class SimplePipelineBuilder(AbstractPipelineBuilder):
    '''
    Builds simple sequential pipelines.
    '''

    def __init__(self):
        self.operators = []

    def add_operator(self, operator):
        self.operators.append(operator)

    def build(self):
        context = Context()
        for i in range(len(self.operators)-1):
            # assemble
            self.operators[i].setup(context, Collector([self.operators[i+1]]))
        self.operators[-1].setup(context, Collector())
        return Pipeline(self.operators[0], self.operators, PipelineController())
