use strict;
use warnings;

use Test::More;

use AI::FANN qw(:all);
use File::Temp;

my $TempFile = File::Temp->new(TEMPLATE => "ai-fann-xor-test-XXXX", SUFFIX => ".ann", UNLINK => 1);

{
    my $ann = AI::FANN->new_standard(2, 3, 1);

    $ann->hidden_activation_function(FANN_SIGMOID_SYMMETRIC);
    $ann->output_activation_function(FANN_SIGMOID_SYMMETRIC);

    # create the training data for a XOR operator:
    my $xor_train = AI::FANN::TrainData->new( [-1, -1], [-1],
                                            [-1, 1], [1],
                                            [1, -1], [1],
                                            [1, 1], [-1] );

    $ann->train_on_data($xor_train, 500000, 1000, 0.001);

    $ann->save($TempFile->filename);
}

{
    my $ann = AI::FANN->new_from_file($TempFile->filename);
    for my $a (-1, 1) {
        for my $b (-1, 1) {
            my $expected = (($a == 1 ? 1 : 0) xor ($b == 1 ? 1 : 0)) == 1 ? 1 : -1;
            my $out = $ann->run([$a, $b]);
            is(scalar(@$out), 1, "Got one output");
            my $got =   $out->[0] >=  0.9 ?  1
                      : $out->[0] <= -0.9 ? -1
                      :                     $out->[0];
            is($got, $expected, "For input [$a, $b] we expect [$expected] or close enough");
        }
    }
}

done_testing;
