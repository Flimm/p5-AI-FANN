use strict;
use warnings;

use Test::More;

use AI::FANN qw(:all);
use File::Temp;

BEGIN {
    eval "use Test::LeakTrace";
    if ($@) {
        plan skip_all => "Test::LeakTrace is needed";
    }
}


my $TempFile = File::Temp->new(TEMPLATE => "ai-fann-xor-test-XXXX", SUFFIX => ".ann", UNLINK => 1);

{
    my $ann = AI::FANN->new_standard(2, 3, 1);

    no_leaks_ok { $ann->hidden_activation_function(FANN_SIGMOID_SYMMETRIC) } "hidden_activation_function";
    no_leaks_ok { $ann->output_activation_function(FANN_SIGMOID_SYMMETRIC) } "output_activation_function";

    # create the training data for a XOR operator:
    my $xor_train = AI::FANN::TrainData->new( [-1, -1], [-1],
                                            [-1, 1], [1],
                                            [1, -1], [1],
                                            [1, 1], [-1] );

    no_leaks_ok { $ann->train_on_data($xor_train, 500000, 1000, 0.001) } "train_on_data";

    no_leaks_ok { $ann->save($TempFile->filename) } "save";
}

{
    my $ann = AI::FANN->new_from_file($TempFile->filename);
    for my $a (-1, 1) {
        for my $b (-1, 1) {
            no_leaks_ok { $ann->run([$a, $b]) } "run";
        }
    }
}

no_leaks_ok { AI::FANN->new_from_file($TempFile->filename) } "new_from_file";

done_testing;
