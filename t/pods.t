#!/usr/bin/perl

use strict;
use Test::More;

if (! $ENV{TEST_AUTHOR}) {
    plan skip_all => 'Author test. Set $ENV{TEST_AUTHOR} to a true value to run';
}

eval "use Test::Pod 1.00";
plan skip_all => "Test::Pod 1.00 required for testing POD" if $@;

all_pod_files_ok( all_pod_files( qw(blib) ) );
