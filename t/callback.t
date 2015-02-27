use strict;
use warnings;

use Test::More;

use AI::FANN qw(:all);

my @data = ([-1, -1], [-1],
            [-1, 1], [1],
            [1, -1], [1],
            [1, 1], [-1]);

is(scalar(keys %AI::FANN::_callback_for_ann), 0, "Zero keys in %callback to start with");

{
    my $ann = AI::FANN->new_standard(2, 3, 1);

    $ann->hidden_activation_function(FANN_SIGMOID_SYMMETRIC);
    $ann->output_activation_function(FANN_SIGMOID_SYMMETRIC);

    my $xor_train = AI::FANN::TrainData->new(@data);

    cmp_ok($ann->_get_struct_addr(), '>=', 0, "_get_struct_addr returns positive number");
    is($ann->_get_struct_addr(), $ann->_get_struct_addr(), "Consecutive calls of _get_struct_addr consistent");

    my $num_called = 0;
    my $last_epoch = undef;

    my $rc_callback = sub {
        $num_called++;

        my ($c_ann, $train_data, $max_epochs, $epoch_between_reports, $desired_error, $epochs) = @_;
        is(scalar(@_), 6, "Callback got 6 arguments");

        $desired_error = sprintf("%.3f", $desired_error);
        is($c_ann                 , undef  , "Callback received ann argument as expected");
        is($train_data            , undef  , "Callback received train_data argument as expected");
        is($max_epochs            , 500000 , "Callback received max_epochs argument as expected");
        is($epoch_between_reports , 1000   , "Callback received epoch_between_reports as expected");
        is($desired_error         , 0.001  , "Callback received desired_error as expected");
        if (defined $last_epoch) {
            cmp_ok($epochs, '>', $last_epoch, "Callback received epochs greater than last recorded");
        }
        else {
            cmp_ok($epochs, '>', 0, "Callback received epochs argument greater than 0");
        }
        $last_epoch //= $epochs;
        return;
    };

    $ann->set_callback($rc_callback);

    is($num_called, 0, "Callback still hasn't been called");

    is_deeply([values %AI::FANN::_callback_for_ann], [$rc_callback], "Callback registered");

    $ann->train_on_data($xor_train, 500000, 1000, 0.001);

    cmp_ok($num_called, '>=', 1, "Callback called at least once");
}

{
    my $ann = AI::FANN->new_standard(2, 3, 1);

    $ann->hidden_activation_function(FANN_SIGMOID_SYMMETRIC);
    $ann->output_activation_function(FANN_SIGMOID_SYMMETRIC);

    my $xor_train = AI::FANN::TrainData->new(@data);

    my $num_called = 0;
    my $last_epoch = undef;

    my $rc_callback = sub {
        $num_called++;
        return -1;
    };

    $ann->set_callback($rc_callback);

    is($num_called, 0, "Callback still hasn't been called");

    $ann->train_on_data($xor_train, 500000, 1000, 0.001);

    is($num_called, 1, "Callback called exactly once");
}

is(scalar(keys %AI::FANN::_callback_for_ann), 0, "Zero keys in %callback at the end");


done_testing;
