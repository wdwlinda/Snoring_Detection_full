import heapq
from collections import defaultdict


class Plugin(object):
    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, trainer):
        raise NotImplementedError


class Monitor(Plugin):

    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):
        '''
        :param running_average:
        :param epoch_average:
        :param smoothing:
        :param precision:数字输出精度
        :param number_format:  数字输出格式
        :param unit:
        '''
        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        #规定了输出格式
        number_format = ':' + number_format
        '''
        在基类 plugin 中,初始化需要传入interval 参数,此处list[(1, 'iteration'), (1, 'epoch')] 
        代表了插件自身实现的的触发time 跟触发时间
        '''
        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')])

        #是否平滑
        self.smoothing = smoothing
        #增量计算均值
        self.with_running_average = running_average
        self.with_epoch_average = epoch_average

        #输出日志的格式
        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]
        if self.with_running_average:
            self.log_iter_fields += [' ({running_avg' + number_format + '}' + unit + ')']
        if self.with_epoch_average:
            self.log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]

    def register(self, trainer):
        self.trainer = trainer
        #在此处注册的时候,给train 的stats 注册当前状态,比如log 的格式等
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)

    def iteration(self, *args):
        #每个iteration 进行的操作
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        #通过_get_value 方法拿到每个插件的值,放入到stats中
        stats['last'] = self._get_value(*args)

        if self.with_epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in
                                         zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                stats['last'] * (1 - self.smoothing)

    def epoch(self, idx):
        #每个epoch 进行的操作
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            #如果需要计算每轮epoch 的精度等,需要 总数/轮数
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
            stats['epoch_stats'] = (0, 0)


class LossMonitor(Monitor):
    stat_name = 'loss'
    def _get_value(self, iteration, input, target, output, loss):
        return loss.item()


class Logger(Plugin):
    alignment = 4
    #不同字段之间的分隔符
    separator = '#' * 80

    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)

        #需要打印的字段,如loss acc
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))
        # 遵循XPath路径的格式。以AccuracyMonitor为例子，如果你想打印所有的状态，
        # 那么你只需要令fields=[AccuracyMonitor.stat_name]，也就是，['accuracy']，
        # 而如果你想只打印AccuracyMonitor的子状态'last'，那么你就只需要设置为
        # ['accuracy.last'],而这里的split当然就是为了获得[['accuracy', 'last']]
        # 这是为了之后的子状态解析（类似XPath路径解析）所使用的。

    def _join_results(self, results):
        # 这个函数主要是将获得的子状态的结果进行组装。
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        print(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        #对其输出格式
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        # 这个函数是核心，负责将查找到的最底层的子模块的结果提取出来。
        output = []
        name = ''
        if isinstance(stat, dict):
            '''
            通过插件的子stat去拿到每一轮的信息,如LOSS等
            '''
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            # 找到自定义的输出名称。y有时候我们并不像打印对应的Key出来，所以可以
            # 在写插件的时候增加多一个'log_name'的键值对，指定打印的名称。默认为
            # field的完整名字。传入的fileds为['accuracy.last']
            # 那么经过初始化之后，fileds=[['accuracy',
            # 'last']]。所以这里的'.'.join(fields)其实是'accuracy.last'。
            # 起到一个还原名称的作用。
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            # 在这里的话，如果子模块stat不是字典且require_dict=False
            # 那么他就会以父模块的打印格式和打印单位作为输出结果的方式。
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        loginfo = []

        if prefix is not None:
            loginfo.append(prefix)
            loginfo.append("\t")

        loginfo.append(output)
        if suffix is not None:
            loginfo.append("\t")
            loginfo.append(suffix)
        self.log("".join(loginfo))

    def iteration(self, *args):
        '''
        :param args:   ( i, batch_input, batch_target,*plugin_data) 的元祖
        :return:
        '''
        self._log_all('log_iter_fields',prefix="iteration:{}".format(args[0]))

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)


class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset

        self.iterations = 0
        self.stats = {}

        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
        '''
        作者将插件的调用进行了分类:
        (1)iteration:一般是在完成一个batch 训练之后进行的事件调用序列（一般不改动网络或者优化器，如：计算准确率）调用序列；
        (2)batch 在进行batch 训练之前需要进行的事件调用序列
        (3)epoch 完成一个epoch 训练之后进行的事件调用序列
        (4)update 完成一个batch训练之后进行的事件(涉及到对网络或者优化器的改动,如:学习率的调整)
        
        iteration 跟update 两种插件调用的时候传入的参数不一样,iteration 会传入batch output,loss 等训练过程中的数据,
        而update传入的的model ,方便对网络的修改
        '''

    def register_plugin(self, plugin):
        #注册插件
        plugin.register(self)

        #插件的触发间隔,一般是这样的形式[(1, 'iteration'), (1, 'epoch')]
        intervals = plugin.trigger_interval

        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            #unit 是事件的触发类别
            queue = self.plugin_queues[unit]
            '''添加事件， 这里的duration就是触发间隔,，以后在调用插件的时候，
            会进行更新  duration 决定了比如在第几个iteration or epoch 触发事件。len(queue)这里应当理解为优先级（越小越高）
            【在相同duration的情况下决定调用的顺序】，根据加入队列的早晚决定。'''
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        #调用插件
        args = (time,) + args
        #这里的time 最基本的意思是次数,如(iteration or epoch)
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            '''如果队列第一个事件的duration（也就是触发时间点）小于当前times'''
            plugin = queue[0][2]
            '''调用相关队列相应的方法，所以如果是继承Plugin类的插件，
                       必须实现 iteration、batch、epoch和update中的至少一个且名字必须一致。'''
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            '''根据插件的事件触发间隔，来更新事件队列里的事件 duration'''
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)
            '''加入新的事件并弹出最小堆的堆头。最小堆重新排序。'''

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            self.train()
            #进行每次epoch 的更新
            self.call_plugins('epoch', i)

    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input, batch_target = data
            #在每次获取batch data 后进行更新
            self.call_plugins('batch', i, batch_input, batch_target)
            input_var = batch_input
            target_var = batch_target
            #这里是给后续插件做缓存部分数据,这里是网络输出与loss
            plugin_data = [None, None]

            def closure():
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

        self.iterations += i


if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset, DataLoader
    from dataset.dataloader import AudioDataset, SimpleAudioDataset
    from models.image_classification import img_classifier
    import train_utils
    from utils import configuration
    ImageClassifier = img_classifier.ImageClassifier
    CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_valid_config.yml'
    data_path = rf''

    config = configuration.load_config(CONFIG_PATH)
    test_dataset = AudioDataset(config, data_path)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.dataset.batch_size, shuffle=config.dataset.shuffle, pin_memory=config.train.pin_memory, num_workers=config.train.num_workers)
    
    net = ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels, activation=config.model.activation,
        out_channels=config.model.out_channels, pretrained=False, dim=1, output_structure=None)

    loss_func = train_utils.get_loss(config.train.loss)    
    if torch.cuda.is_available():
        net.cuda()
    optimizer = train_utils.create_optimizer(config.optimizer_config, net)


    trainer = Trainer(pretrainModel=net,criterion=loss_func,optimizer=optimizer,train_loader=test_dataloader)

    trainer.register_plugin(LossMonitor())

    trainer.register_plugin(Logger(['loss']))
   
    trainer.run(50)