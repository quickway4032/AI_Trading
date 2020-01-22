#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from textclassifier import TextClassifier

epochs = 2
batch_size = 512
sequence_length = 40
learning_rate = 1e-3
clip = 5
best_val_acc = 0
print_every = 100

class Calculator:
    
    def __init__(self,
                 input_directory,
                 output_directory,
                 cache_directory,
                ):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.cache_directory = cache_directory
 
        self.cfg = gic.util.get_config()
                    
        logger.debug(self.cfg)
        
     def model_build(self):
         
         model = TextClassifier(len(vocab), 10, 6, 5, dropout=0.1, lstm_layers=2)
         model.embedding.weight.data.uniform_(-1, 1)
         input = torch.randint(0, 1000, (5, 4), dtype=torch.int64)
         hidden = model.init_hidden(4)

logps, _ = model.forward(input, hidden)

valid_split = int(0.9*len(token_ids))

train_features = token_ids[:valid_split]
valid_features = token_ids[valid_split:]
train_labels = sentiments[:valid_split]
valid_labels = sentiments[valid_split:]

text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=sequence_length, batch_size=batch_size)))
model = TextClassifier(len(vocab)+1, 200, 128, clip, dropout=0.)
hidden = model.init_hidden(batch_size)
logps, hidden = model.forward(text_batch, hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassifier(len(vocab)+1, 1024, batch_size, clip, lstm_layers=2, dropout=0.2)
model.embedding.weight.data.uniform_(-1, 1)
model.to(device)


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in range(epochs):
    print('Starting epoch {}'.format(epoch + 1))
    steps = 0        
    hidden = model.init_hidden(batch_size)
    for text_batch, labels in dataloader(
            train_features, train_labels, batch_size=batch_size, sequence_length=sequence_length, shuffle=True):
        if text_batch.size() != torch.Size([sequence_length, batch_size]):
            continue
        steps += 1
        hidden = tuple([each.data for each in hidden])
        
        # Set Device
        text_batch, labels = text_batch.to(device), labels.to(device)
        for each in hidden:
            each.to(device)
        
        model.zero_grad()
        log_ps, hidden = model.forward(text_batch, hidden)
        loss = criterion(log_ps, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if steps % print_every == 0:
            model.eval()
            val_losses = []
            val_accuracy = []
            val_hidden = model.init_hidden(batch_size)

            for val_text_batch, val_labels in dataloader(
            valid_features, valid_labels, batch_size=batch_size, sequence_length=sequence_length):
                if val_text_batch.size() != torch.Size([sequence_length, batch_size]):
                    continue
                val_text_batch, val_labels = val_text_batch.to(device), val_labels.to(device)
                val_hidden = tuple([each.data for each in val_hidden])
                for each in val_hidden:
                    each.to(device)
                val_log_ps, hidden = model.forward(val_text_batch, val_hidden)
                val_loss = criterion(val_log_ps.squeeze(), val_labels)
                val_losses.append(val_loss.item())
                
                val_ps = torch.exp(val_log_ps)
                top_p, top_class = val_ps.topk(1, dim=1)
                equals = top_class == val_labels.view(*top_class.shape)
                val_accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())

            model.train()
            this_val_acc = sum(val_accuracy)/len(val_accuracy)
            
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(steps),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(sum(val_losses)/len(val_losses)),
                 "Val Accuracy: {:.4f}".format(this_val_acc))
            if this_val_acc > best_val_acc:
                torch.save({
            'epoch': epoch,
            'step': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'best_model')
                best_val_acc = this_val_acc
                print("New best accuracy - model saved")
        
        
        
    
    def forward_selection(self, gic_type, gic_name, data):
        
        logger.debug(f'Variable forward selection for {gic_type}/{gic_name}')
        if gic_type == 'BNSGICs': 
            df = gic.util.get_gic_from_edw(data, gic_name)
            a =  gic.estimate_model_with_sampling.ForwardSelectionBNS(gic_name, 
                                                                    df, 
                                                                    self.input_directory,
                                                                    self.output_directory ,                                                
                                                                    )
            a.run()
        elif gic_type == 'TNGGICs':
            df = gic.util.get_gic_from_edl(data, gic_name, is_ip = False, is_tng = True)
            a =  gic.estimate_model_with_sampling.ForwardSelectionTNG(gic_name, 
                                                                    df, 
                                                                    self.input_directory,
                                                                    self.output_directory ,                                                
                                                                    )
            a.run()
        
        
    def estimate(self, gic_type, gic_name, data, parameter_directory):
        
        logger.debug(f'Estimating final model for {gic_type}/{gic_name}')
        
        '''Estimate model for full redemption rate'''
        if gic_type == 'BNSGICs': 
            df = gic.util.get_gic_from_edw(data, gic_name)
        elif gic_type == 'TNGGICs':
            df = gic.util.get_gic_from_edl(data, gic_name, is_ip = False, is_tng = True)
            
        a =  gic.estimate_model.EstimateFinalModel(gic_name, 
                                                    df, 
                                                    self.input_directory,
                                                    self.output_directory ,
                                                    parameter_directory
                                                    )
        a.run()
        
    def forecast(self, gic_type, gic_name, data, start_date, end_date, parameter_directory):
        
        if gic_type == 'BNSGICs':
            df = gic.util.get_gic_from_edl(data, gic_name, is_ip = False, is_tng = False)
           
        elif gic_type == 'TNGGICs':
            df = gic.util.get_gic_from_edl(data, gic_name, is_ip = False, is_tng = True)
        f = gic.forecast_generator.Forecast(gic_name, 
                                             df, 
                                             start_date, 
                                             end_date, 
                                             parameter_directory,
                                             self.input_directory,
                                             self.output_directory)
        f.run()
        
    def run(self, data_function, run_function):
        
        (self.input_directory).mkdir(parents=True, exist_ok=True)
        (self.output_directory/'estimate'/'forward_selection_results').mkdir(parents=True, exist_ok=True)
        (self.output_directory/'estimate'/'final_model').mkdir(parents=True, exist_ok=True)  
        (self.output_directory/'params').mkdir(parents=True, exist_ok=True)  
        (self.output_directory/'forecast').mkdir(parents=True, exist_ok=True)
        (self.output_directory/'backtest').mkdir(parents=True, exist_ok=True)
        
        gic_type = 'BNSGICs'
        data = data_function(gic_type)
        gic_plan = self.cfg['gic_plan']
        for gic_name in gic_plan:
            run_function(gic_type, gic_name, data)
            
        gic_type = 'TNGGICs'
        data = data_function(gic_type)
        gic_plan = self.cfg['tng_gic_plan']
        for gic_name in gic_plan:
            run_function(gic_type, gic_name, data)
    
def forward_selection(calc, args):
    calc.run(calc.read_data,
             functools.partial(calc.forward_selection))
                   
def estimate(calc, args):
    calc.run(calc.read_data, 
                 functools.partial(calc.estimate,
                                   parameter_directory=args.model_variable_directory))

    for p in (args.output_directory/'estimate'/'final_model').glob('*/model_summary_2.csv'):
        shutil.copy2(p, args.output_directory/'params'/f'{p.parent.name}.csv')
    for p in (args.output_directory/'estimate'/'final_model').glob('*/rolling_average_partial_ERR.csv'):
        shutil.copy2(p, args.output_directory/'params'/f'{p.parent.name}_partial_ERR.csv')

    ci = gic.update_ci_for_backtest.CI(args.input_directory,
                                        args.output_directory,
                                        args.cache_directory,
                                        args.output_directory/'params')
    ci.run()

    for p in (args.output_directory/'backtest').glob('*/rate_ci_1.csv'):
        shutil.copy2(p, args.output_directory/'params'/f'{p.parent.name}_rate_ci_1.csv')
    for p in (args.output_directory/'backtest').glob('*/bal_ci_1.csv'):
        shutil.copy2(p, args.output_directory/'params'/f'{p.parent.name}_bal_ci_1.csv')

def promote(calc, args):
    for p in (args.output_directory/'params').glob('*.csv'):
        mrmalm.run(['add_param', str(p)])
            
def forecast(calc, args):
    calc.run(calc.read_backtest_data,
             functools.partial(calc.forecast,
                               start_date=args.start_date,
                               end_date=args.end_date,
                               parameter_directory=args.parameter_directory))
            
def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_directory', default=pathlib.Path('./input'),
                        type=lambda d: pathlib.Path(d),
                        nargs='?', help='Input directory')
    parser.add_argument('--output_directory', default=pathlib.Path('./output'),
                        type=lambda d: pathlib.Path(d),
                        nargs='?', help='Output directory')
    parser.add_argument('--cache_directory', default=gic.data.insight.table.default_cache_directory,
                        type=lambda d: pathlib.Path(d),
                        nargs='?', help='Cache directory')
    parser.add_argument('--static_directory', default=pathlib.Path('./input/static'),
                        type=lambda d: pathlib.Path(d),
                        nargs='?', help='Static directory')    
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help=('Verbose (add output);  can be specificed '
                              'multiple times to increase verbosity'))

    sub_parsers = parser.add_subparsers()
    
    forward_selection_parser = sub_parsers.add_parser('Forward_selection')
    forward_selection_parser.set_defaults(func=forward_selection)
    
    estimate_parser = sub_parsers.add_parser('Estimate')
    estimate_parser.set_defaults(func=estimate)
    estimate_parser.add_argument('--model_variable_directory', 
                           default=pathlib.Path('./output/estimate/forward_selection_results'),
                           type=lambda d: pathlib.Path(d),
                           nargs='?', help='Model variables directory from forward selection')   
    estimate_parser.add_argument('--promote', action='store_true', default=False, help='promote generated estimates')

    forecast_parser = sub_parsers.add_parser('Forecast')
    forecast_parser.set_defaults(func=forecast)
    forecast_parser.add_argument('--start_date',
                        default=(pandas.Timestamp.today().date() - pandas.tseries.offsets.MonthEnd()),
                        type=lambda x: pandas.Timestamp(x),
                        help='Calibration date in YYYY-MM-DD format')
    forecast_parser.add_argument('--end_date',
                        default=(pandas.Timestamp.today().date() - pandas.tseries.offsets.MonthEnd()),
                        type=lambda x: pandas.Timestamp(x),
                        help='Calibration date in YYYY-MM-DD format')
    forecast_parser.add_argument('--parameter_directory', 
                           default=pathlib.Path('./params'),
                           type=lambda d: pathlib.Path(d),
                           nargs='?', help='Model result directory')   

    forecast_parser = sub_parsers.add_parser('Promote')
    forecast_parser.set_defaults(func=promote)

    return parser.parse_args(args)

def run(args):
    c = Calculator(args.input_directory,
                   args.output_directory,
                   args.cache_directory
                   )
    
    try:
        args.func(c, args)
    except RuntimeError as err:
        logger.error(traceback.format_exc())
        raise err

def main(args=None):    
    lf = logging.Formatter(('%(asctime)s:[%(levelname)s]:[%(name)s]:%(message)s'))
    lh = logging.StreamHandler()
    lh.setFormatter(lf)
    logger.addHandler(lh)

    args = parse(args)

    # Add a file logger
    (args.output_directory/'logs').mkdir(parents=True, exist_ok=True)
    lh = logging.FileHandler(args.output_directory/'logs'/'log.txt')
    lh.setFormatter(lf)
    logger.addHandler(lh)

    if 0 == args.verbose:
        pass
    elif 1 == args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    
    try:
        run(args)
    finally:
        # Delete handler to allow archiving to take place
        # (otherwise, there is an open file)
        logger.removeHandler(lh)

if __name__ == '__main__':
    main()