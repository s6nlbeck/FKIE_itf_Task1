??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
}
dense_300/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*!
shared_namedense_300/kernel
v
$dense_300/kernel/Read/ReadVariableOpReadVariableOpdense_300/kernel*
_output_shapes
:	?@*
dtype0
t
dense_300/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_300/bias
m
"dense_300/bias/Read/ReadVariableOpReadVariableOpdense_300/bias*
_output_shapes
:@*
dtype0
|
dense_301/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_301/kernel
u
$dense_301/kernel/Read/ReadVariableOpReadVariableOpdense_301/kernel*
_output_shapes

:@@*
dtype0
t
dense_301/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_301/bias
m
"dense_301/bias/Read/ReadVariableOpReadVariableOpdense_301/bias*
_output_shapes
:@*
dtype0
|
dense_302/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_302/kernel
u
$dense_302/kernel/Read/ReadVariableOpReadVariableOpdense_302/kernel*
_output_shapes

:@@*
dtype0
t
dense_302/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_302/bias
m
"dense_302/bias/Read/ReadVariableOpReadVariableOpdense_302/bias*
_output_shapes
:@*
dtype0
|
dense_303/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_303/kernel
u
$dense_303/kernel/Read/ReadVariableOpReadVariableOpdense_303/kernel*
_output_shapes

:@@*
dtype0
t
dense_303/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_303/bias
m
"dense_303/bias/Read/ReadVariableOpReadVariableOpdense_303/bias*
_output_shapes
:@*
dtype0
|
dense_304/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_304/kernel
u
$dense_304/kernel/Read/ReadVariableOpReadVariableOpdense_304/kernel*
_output_shapes

:@*
dtype0
t
dense_304/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_304/bias
m
"dense_304/bias/Read/ReadVariableOpReadVariableOpdense_304/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/dense_300/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*(
shared_nameAdam/dense_300/kernel/m
?
+Adam/dense_300/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_300/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_300/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_300/bias/m
{
)Adam/dense_300/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_300/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_301/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_301/kernel/m
?
+Adam/dense_301/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_301/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_301/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_301/bias/m
{
)Adam/dense_301/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_301/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_302/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_302/kernel/m
?
+Adam/dense_302/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_302/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_302/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_302/bias/m
{
)Adam/dense_302/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_302/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_303/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_303/kernel/m
?
+Adam/dense_303/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_303/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_303/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_303/bias/m
{
)Adam/dense_303/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_303/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_304/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_304/kernel/m
?
+Adam/dense_304/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_304/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_304/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_304/bias/m
{
)Adam/dense_304/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_304/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_300/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*(
shared_nameAdam/dense_300/kernel/v
?
+Adam/dense_300/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_300/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_300/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_300/bias/v
{
)Adam/dense_300/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_300/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_301/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_301/kernel/v
?
+Adam/dense_301/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_301/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_301/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_301/bias/v
{
)Adam/dense_301/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_301/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_302/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_302/kernel/v
?
+Adam/dense_302/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_302/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_302/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_302/bias/v
{
)Adam/dense_302/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_302/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_303/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_303/kernel/v
?
+Adam/dense_303/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_303/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_303/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_303/bias/v
{
)Adam/dense_303/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_303/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_304/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_304/kernel/v
?
+Adam/dense_304/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_304/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_304/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_304/bias/v
{
)Adam/dense_304/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_304/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?8
value?8B?8 B?8
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?
+iter

,beta_1

-beta_2
	.decay
/learning_ratem_m`mambmcmdme mf%mg&mhvivjvkvlvmvnvo vp%vq&vr
 
F
0
1
2
3
4
5
6
 7
%8
&9
F
0
1
2
3
4
5
6
 7
%8
&9
?
0layer_regularization_losses
1non_trainable_variables
regularization_losses

2layers
3layer_metrics
	trainable_variables
4metrics

	variables
 
\Z
VARIABLE_VALUEdense_300/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_300/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
5layer_regularization_losses
6non_trainable_variables

7layers
8layer_metrics
trainable_variables
	variables
9metrics
regularization_losses
\Z
VARIABLE_VALUEdense_301/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_301/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
:layer_regularization_losses
;non_trainable_variables

<layers
=layer_metrics
trainable_variables
	variables
>metrics
regularization_losses
\Z
VARIABLE_VALUEdense_302/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_302/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?layer_regularization_losses
@non_trainable_variables

Alayers
Blayer_metrics
trainable_variables
	variables
Cmetrics
regularization_losses
\Z
VARIABLE_VALUEdense_303/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_303/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
Dlayer_regularization_losses
Enon_trainable_variables

Flayers
Glayer_metrics
!trainable_variables
"	variables
Hmetrics
#regularization_losses
\Z
VARIABLE_VALUEdense_304/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_304/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
Ilayer_regularization_losses
Jnon_trainable_variables

Klayers
Llayer_metrics
'trainable_variables
(	variables
Mmetrics
)regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
*
0
1
2
3
4
5
 

N0
O1
P2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Qtotal
	Rcount
S	variables
T	keras_api
D
	Utotal
	Vcount
W
_fn_kwargs
X	variables
Y	keras_api
D
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1

S	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

X	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

]	variables
}
VARIABLE_VALUEAdam/dense_300/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_300/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_301/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_301/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_302/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_302/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_303/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_303/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_304/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_304/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_300/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_300/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_301/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_301/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_302/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_302/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_303/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_303/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_304/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_304/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
serving_default_input_61Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_61dense_300/kerneldense_300/biasdense_301/kerneldense_301/biasdense_302/kerneldense_302/biasdense_303/kerneldense_303/biasdense_304/kerneldense_304/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1967142
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_300/kernel/Read/ReadVariableOp"dense_300/bias/Read/ReadVariableOp$dense_301/kernel/Read/ReadVariableOp"dense_301/bias/Read/ReadVariableOp$dense_302/kernel/Read/ReadVariableOp"dense_302/bias/Read/ReadVariableOp$dense_303/kernel/Read/ReadVariableOp"dense_303/bias/Read/ReadVariableOp$dense_304/kernel/Read/ReadVariableOp"dense_304/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/dense_300/kernel/m/Read/ReadVariableOp)Adam/dense_300/bias/m/Read/ReadVariableOp+Adam/dense_301/kernel/m/Read/ReadVariableOp)Adam/dense_301/bias/m/Read/ReadVariableOp+Adam/dense_302/kernel/m/Read/ReadVariableOp)Adam/dense_302/bias/m/Read/ReadVariableOp+Adam/dense_303/kernel/m/Read/ReadVariableOp)Adam/dense_303/bias/m/Read/ReadVariableOp+Adam/dense_304/kernel/m/Read/ReadVariableOp)Adam/dense_304/bias/m/Read/ReadVariableOp+Adam/dense_300/kernel/v/Read/ReadVariableOp)Adam/dense_300/bias/v/Read/ReadVariableOp+Adam/dense_301/kernel/v/Read/ReadVariableOp)Adam/dense_301/bias/v/Read/ReadVariableOp+Adam/dense_302/kernel/v/Read/ReadVariableOp)Adam/dense_302/bias/v/Read/ReadVariableOp+Adam/dense_303/kernel/v/Read/ReadVariableOp)Adam/dense_303/bias/v/Read/ReadVariableOp+Adam/dense_304/kernel/v/Read/ReadVariableOp)Adam/dense_304/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1967516
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_300/kerneldense_300/biasdense_301/kerneldense_301/biasdense_302/kerneldense_302/biasdense_303/kerneldense_303/biasdense_304/kerneldense_304/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/dense_300/kernel/mAdam/dense_300/bias/mAdam/dense_301/kernel/mAdam/dense_301/bias/mAdam/dense_302/kernel/mAdam/dense_302/bias/mAdam/dense_303/kernel/mAdam/dense_303/bias/mAdam/dense_304/kernel/mAdam/dense_304/bias/mAdam/dense_300/kernel/vAdam/dense_300/bias/vAdam/dense_301/kernel/vAdam/dense_301/bias/vAdam/dense_302/kernel/vAdam/dense_302/bias/vAdam/dense_303/kernel/vAdam/dense_303/bias/vAdam/dense_304/kernel/vAdam/dense_304/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1967649??
?
?
E__inference_model_60_layer_call_and_return_conditional_losses_1966998
input_61
dense_300_1966972
dense_300_1966974
dense_301_1966977
dense_301_1966979
dense_302_1966982
dense_302_1966984
dense_303_1966987
dense_303_1966989
dense_304_1966992
dense_304_1966994
identity??!dense_300/StatefulPartitionedCall?!dense_301/StatefulPartitionedCall?!dense_302/StatefulPartitionedCall?!dense_303/StatefulPartitionedCall?!dense_304/StatefulPartitionedCall?
!dense_300/StatefulPartitionedCallStatefulPartitionedCallinput_61dense_300_1966972dense_300_1966974*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_300_layer_call_and_return_conditional_losses_19668442#
!dense_300/StatefulPartitionedCall?
!dense_301/StatefulPartitionedCallStatefulPartitionedCall*dense_300/StatefulPartitionedCall:output:0dense_301_1966977dense_301_1966979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_301_layer_call_and_return_conditional_losses_19668712#
!dense_301/StatefulPartitionedCall?
!dense_302/StatefulPartitionedCallStatefulPartitionedCall*dense_301/StatefulPartitionedCall:output:0dense_302_1966982dense_302_1966984*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_302_layer_call_and_return_conditional_losses_19668982#
!dense_302/StatefulPartitionedCall?
!dense_303/StatefulPartitionedCallStatefulPartitionedCall*dense_302/StatefulPartitionedCall:output:0dense_303_1966987dense_303_1966989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_303_layer_call_and_return_conditional_losses_19669252#
!dense_303/StatefulPartitionedCall?
!dense_304/StatefulPartitionedCallStatefulPartitionedCall*dense_303/StatefulPartitionedCall:output:0dense_304_1966992dense_304_1966994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_304_layer_call_and_return_conditional_losses_19669522#
!dense_304/StatefulPartitionedCall?
IdentityIdentity*dense_304/StatefulPartitionedCall:output:0"^dense_300/StatefulPartitionedCall"^dense_301/StatefulPartitionedCall"^dense_302/StatefulPartitionedCall"^dense_303/StatefulPartitionedCall"^dense_304/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_300/StatefulPartitionedCall!dense_300/StatefulPartitionedCall2F
!dense_301/StatefulPartitionedCall!dense_301/StatefulPartitionedCall2F
!dense_302/StatefulPartitionedCall!dense_302/StatefulPartitionedCall2F
!dense_303/StatefulPartitionedCall!dense_303/StatefulPartitionedCall2F
!dense_304/StatefulPartitionedCall!dense_304/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_61
?
?
*__inference_model_60_layer_call_fn_1967270

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_19670842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_300_layer_call_fn_1967290

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_300_layer_call_and_return_conditional_losses_19668442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_304_layer_call_and_return_conditional_losses_1967361

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_model_60_layer_call_fn_1967245

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_19670302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_301_layer_call_fn_1967310

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_301_layer_call_and_return_conditional_losses_19668712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?T
?
 __inference__traced_save_1967516
file_prefix/
+savev2_dense_300_kernel_read_readvariableop-
)savev2_dense_300_bias_read_readvariableop/
+savev2_dense_301_kernel_read_readvariableop-
)savev2_dense_301_bias_read_readvariableop/
+savev2_dense_302_kernel_read_readvariableop-
)savev2_dense_302_bias_read_readvariableop/
+savev2_dense_303_kernel_read_readvariableop-
)savev2_dense_303_bias_read_readvariableop/
+savev2_dense_304_kernel_read_readvariableop-
)savev2_dense_304_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_dense_300_kernel_m_read_readvariableop4
0savev2_adam_dense_300_bias_m_read_readvariableop6
2savev2_adam_dense_301_kernel_m_read_readvariableop4
0savev2_adam_dense_301_bias_m_read_readvariableop6
2savev2_adam_dense_302_kernel_m_read_readvariableop4
0savev2_adam_dense_302_bias_m_read_readvariableop6
2savev2_adam_dense_303_kernel_m_read_readvariableop4
0savev2_adam_dense_303_bias_m_read_readvariableop6
2savev2_adam_dense_304_kernel_m_read_readvariableop4
0savev2_adam_dense_304_bias_m_read_readvariableop6
2savev2_adam_dense_300_kernel_v_read_readvariableop4
0savev2_adam_dense_300_bias_v_read_readvariableop6
2savev2_adam_dense_301_kernel_v_read_readvariableop4
0savev2_adam_dense_301_bias_v_read_readvariableop6
2savev2_adam_dense_302_kernel_v_read_readvariableop4
0savev2_adam_dense_302_bias_v_read_readvariableop6
2savev2_adam_dense_303_kernel_v_read_readvariableop4
0savev2_adam_dense_303_bias_v_read_readvariableop6
2savev2_adam_dense_304_kernel_v_read_readvariableop4
0savev2_adam_dense_304_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_300_kernel_read_readvariableop)savev2_dense_300_bias_read_readvariableop+savev2_dense_301_kernel_read_readvariableop)savev2_dense_301_bias_read_readvariableop+savev2_dense_302_kernel_read_readvariableop)savev2_dense_302_bias_read_readvariableop+savev2_dense_303_kernel_read_readvariableop)savev2_dense_303_bias_read_readvariableop+savev2_dense_304_kernel_read_readvariableop)savev2_dense_304_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_dense_300_kernel_m_read_readvariableop0savev2_adam_dense_300_bias_m_read_readvariableop2savev2_adam_dense_301_kernel_m_read_readvariableop0savev2_adam_dense_301_bias_m_read_readvariableop2savev2_adam_dense_302_kernel_m_read_readvariableop0savev2_adam_dense_302_bias_m_read_readvariableop2savev2_adam_dense_303_kernel_m_read_readvariableop0savev2_adam_dense_303_bias_m_read_readvariableop2savev2_adam_dense_304_kernel_m_read_readvariableop0savev2_adam_dense_304_bias_m_read_readvariableop2savev2_adam_dense_300_kernel_v_read_readvariableop0savev2_adam_dense_300_bias_v_read_readvariableop2savev2_adam_dense_301_kernel_v_read_readvariableop0savev2_adam_dense_301_bias_v_read_readvariableop2savev2_adam_dense_302_kernel_v_read_readvariableop0savev2_adam_dense_302_bias_v_read_readvariableop2savev2_adam_dense_303_kernel_v_read_readvariableop0savev2_adam_dense_303_bias_v_read_readvariableop2savev2_adam_dense_304_kernel_v_read_readvariableop0savev2_adam_dense_304_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?@:@:@@:@:@@:@:@@:@:@:: : : : : : : : : : : :	?@:@:@@:@:@@:@:@@:@:@::	?@:@:@@:@:@@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::% !

_output_shapes
:	?@: !

_output_shapes
:@:$" 

_output_shapes

:@@: #

_output_shapes
:@:$$ 

_output_shapes

:@@: %

_output_shapes
:@:$& 

_output_shapes

:@@: '

_output_shapes
:@:$( 

_output_shapes

:@: )

_output_shapes
::*

_output_shapes
: 
?	
?
F__inference_dense_303_layer_call_and_return_conditional_losses_1966925

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
F__inference_dense_301_layer_call_and_return_conditional_losses_1966871

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_model_60_layer_call_and_return_conditional_losses_1967084

inputs
dense_300_1967058
dense_300_1967060
dense_301_1967063
dense_301_1967065
dense_302_1967068
dense_302_1967070
dense_303_1967073
dense_303_1967075
dense_304_1967078
dense_304_1967080
identity??!dense_300/StatefulPartitionedCall?!dense_301/StatefulPartitionedCall?!dense_302/StatefulPartitionedCall?!dense_303/StatefulPartitionedCall?!dense_304/StatefulPartitionedCall?
!dense_300/StatefulPartitionedCallStatefulPartitionedCallinputsdense_300_1967058dense_300_1967060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_300_layer_call_and_return_conditional_losses_19668442#
!dense_300/StatefulPartitionedCall?
!dense_301/StatefulPartitionedCallStatefulPartitionedCall*dense_300/StatefulPartitionedCall:output:0dense_301_1967063dense_301_1967065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_301_layer_call_and_return_conditional_losses_19668712#
!dense_301/StatefulPartitionedCall?
!dense_302/StatefulPartitionedCallStatefulPartitionedCall*dense_301/StatefulPartitionedCall:output:0dense_302_1967068dense_302_1967070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_302_layer_call_and_return_conditional_losses_19668982#
!dense_302/StatefulPartitionedCall?
!dense_303/StatefulPartitionedCallStatefulPartitionedCall*dense_302/StatefulPartitionedCall:output:0dense_303_1967073dense_303_1967075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_303_layer_call_and_return_conditional_losses_19669252#
!dense_303/StatefulPartitionedCall?
!dense_304/StatefulPartitionedCallStatefulPartitionedCall*dense_303/StatefulPartitionedCall:output:0dense_304_1967078dense_304_1967080*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_304_layer_call_and_return_conditional_losses_19669522#
!dense_304/StatefulPartitionedCall?
IdentityIdentity*dense_304/StatefulPartitionedCall:output:0"^dense_300/StatefulPartitionedCall"^dense_301/StatefulPartitionedCall"^dense_302/StatefulPartitionedCall"^dense_303/StatefulPartitionedCall"^dense_304/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_300/StatefulPartitionedCall!dense_300/StatefulPartitionedCall2F
!dense_301/StatefulPartitionedCall!dense_301/StatefulPartitionedCall2F
!dense_302/StatefulPartitionedCall!dense_302/StatefulPartitionedCall2F
!dense_303/StatefulPartitionedCall!dense_303/StatefulPartitionedCall2F
!dense_304/StatefulPartitionedCall!dense_304/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_302_layer_call_fn_1967330

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_302_layer_call_and_return_conditional_losses_19668982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_model_60_layer_call_fn_1967107
input_61
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_19670842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_61
?	
?
F__inference_dense_302_layer_call_and_return_conditional_losses_1966898

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
F__inference_dense_302_layer_call_and_return_conditional_losses_1967321

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1967142
input_61
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_19668292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_61
?
?
*__inference_model_60_layer_call_fn_1967053
input_61
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_19670302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_61
?	
?
F__inference_dense_304_layer_call_and_return_conditional_losses_1966952

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
ͬ
?
#__inference__traced_restore_1967649
file_prefix%
!assignvariableop_dense_300_kernel%
!assignvariableop_1_dense_300_bias'
#assignvariableop_2_dense_301_kernel%
!assignvariableop_3_dense_301_bias'
#assignvariableop_4_dense_302_kernel%
!assignvariableop_5_dense_302_bias'
#assignvariableop_6_dense_303_kernel%
!assignvariableop_7_dense_303_bias'
#assignvariableop_8_dense_304_kernel%
!assignvariableop_9_dense_304_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1
assignvariableop_19_total_2
assignvariableop_20_count_2/
+assignvariableop_21_adam_dense_300_kernel_m-
)assignvariableop_22_adam_dense_300_bias_m/
+assignvariableop_23_adam_dense_301_kernel_m-
)assignvariableop_24_adam_dense_301_bias_m/
+assignvariableop_25_adam_dense_302_kernel_m-
)assignvariableop_26_adam_dense_302_bias_m/
+assignvariableop_27_adam_dense_303_kernel_m-
)assignvariableop_28_adam_dense_303_bias_m/
+assignvariableop_29_adam_dense_304_kernel_m-
)assignvariableop_30_adam_dense_304_bias_m/
+assignvariableop_31_adam_dense_300_kernel_v-
)assignvariableop_32_adam_dense_300_bias_v/
+assignvariableop_33_adam_dense_301_kernel_v-
)assignvariableop_34_adam_dense_301_bias_v/
+assignvariableop_35_adam_dense_302_kernel_v-
)assignvariableop_36_adam_dense_302_bias_v/
+assignvariableop_37_adam_dense_303_kernel_v-
)assignvariableop_38_adam_dense_303_bias_v/
+assignvariableop_39_adam_dense_304_kernel_v-
)assignvariableop_40_adam_dense_304_bias_v
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_300_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_300_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_301_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_301_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_302_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_302_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_303_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_303_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_304_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_304_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_300_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_300_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_301_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_301_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_302_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_302_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_303_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_303_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_304_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_304_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_300_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_300_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_301_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_301_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_302_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_302_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_303_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_303_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_304_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_304_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41?
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
F__inference_dense_303_layer_call_and_return_conditional_losses_1967341

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?1
?
E__inference_model_60_layer_call_and_return_conditional_losses_1967181

inputs,
(dense_300_matmul_readvariableop_resource-
)dense_300_biasadd_readvariableop_resource,
(dense_301_matmul_readvariableop_resource-
)dense_301_biasadd_readvariableop_resource,
(dense_302_matmul_readvariableop_resource-
)dense_302_biasadd_readvariableop_resource,
(dense_303_matmul_readvariableop_resource-
)dense_303_biasadd_readvariableop_resource,
(dense_304_matmul_readvariableop_resource-
)dense_304_biasadd_readvariableop_resource
identity?? dense_300/BiasAdd/ReadVariableOp?dense_300/MatMul/ReadVariableOp? dense_301/BiasAdd/ReadVariableOp?dense_301/MatMul/ReadVariableOp? dense_302/BiasAdd/ReadVariableOp?dense_302/MatMul/ReadVariableOp? dense_303/BiasAdd/ReadVariableOp?dense_303/MatMul/ReadVariableOp? dense_304/BiasAdd/ReadVariableOp?dense_304/MatMul/ReadVariableOp?
dense_300/MatMul/ReadVariableOpReadVariableOp(dense_300_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02!
dense_300/MatMul/ReadVariableOp?
dense_300/MatMulMatMulinputs'dense_300/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_300/MatMul?
 dense_300/BiasAdd/ReadVariableOpReadVariableOp)dense_300_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_300/BiasAdd/ReadVariableOp?
dense_300/BiasAddBiasAdddense_300/MatMul:product:0(dense_300/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_300/BiasAdd
dense_300/SigmoidSigmoiddense_300/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_300/Sigmoid?
dense_301/MatMul/ReadVariableOpReadVariableOp(dense_301_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_301/MatMul/ReadVariableOp?
dense_301/MatMulMatMuldense_300/Sigmoid:y:0'dense_301/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_301/MatMul?
 dense_301/BiasAdd/ReadVariableOpReadVariableOp)dense_301_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_301/BiasAdd/ReadVariableOp?
dense_301/BiasAddBiasAdddense_301/MatMul:product:0(dense_301/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_301/BiasAdd
dense_301/SigmoidSigmoiddense_301/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_301/Sigmoid?
dense_302/MatMul/ReadVariableOpReadVariableOp(dense_302_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_302/MatMul/ReadVariableOp?
dense_302/MatMulMatMuldense_301/Sigmoid:y:0'dense_302/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_302/MatMul?
 dense_302/BiasAdd/ReadVariableOpReadVariableOp)dense_302_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_302/BiasAdd/ReadVariableOp?
dense_302/BiasAddBiasAdddense_302/MatMul:product:0(dense_302/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_302/BiasAdd
dense_302/SigmoidSigmoiddense_302/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_302/Sigmoid?
dense_303/MatMul/ReadVariableOpReadVariableOp(dense_303_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_303/MatMul/ReadVariableOp?
dense_303/MatMulMatMuldense_302/Sigmoid:y:0'dense_303/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_303/MatMul?
 dense_303/BiasAdd/ReadVariableOpReadVariableOp)dense_303_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_303/BiasAdd/ReadVariableOp?
dense_303/BiasAddBiasAdddense_303/MatMul:product:0(dense_303/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_303/BiasAdd
dense_303/SigmoidSigmoiddense_303/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_303/Sigmoid?
dense_304/MatMul/ReadVariableOpReadVariableOp(dense_304_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_304/MatMul/ReadVariableOp?
dense_304/MatMulMatMuldense_303/Sigmoid:y:0'dense_304/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_304/MatMul?
 dense_304/BiasAdd/ReadVariableOpReadVariableOp)dense_304_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_304/BiasAdd/ReadVariableOp?
dense_304/BiasAddBiasAdddense_304/MatMul:product:0(dense_304/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_304/BiasAdd
dense_304/SigmoidSigmoiddense_304/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_304/Sigmoid?
IdentityIdentitydense_304/Sigmoid:y:0!^dense_300/BiasAdd/ReadVariableOp ^dense_300/MatMul/ReadVariableOp!^dense_301/BiasAdd/ReadVariableOp ^dense_301/MatMul/ReadVariableOp!^dense_302/BiasAdd/ReadVariableOp ^dense_302/MatMul/ReadVariableOp!^dense_303/BiasAdd/ReadVariableOp ^dense_303/MatMul/ReadVariableOp!^dense_304/BiasAdd/ReadVariableOp ^dense_304/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2D
 dense_300/BiasAdd/ReadVariableOp dense_300/BiasAdd/ReadVariableOp2B
dense_300/MatMul/ReadVariableOpdense_300/MatMul/ReadVariableOp2D
 dense_301/BiasAdd/ReadVariableOp dense_301/BiasAdd/ReadVariableOp2B
dense_301/MatMul/ReadVariableOpdense_301/MatMul/ReadVariableOp2D
 dense_302/BiasAdd/ReadVariableOp dense_302/BiasAdd/ReadVariableOp2B
dense_302/MatMul/ReadVariableOpdense_302/MatMul/ReadVariableOp2D
 dense_303/BiasAdd/ReadVariableOp dense_303/BiasAdd/ReadVariableOp2B
dense_303/MatMul/ReadVariableOpdense_303/MatMul/ReadVariableOp2D
 dense_304/BiasAdd/ReadVariableOp dense_304/BiasAdd/ReadVariableOp2B
dense_304/MatMul/ReadVariableOpdense_304/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_model_60_layer_call_and_return_conditional_losses_1966969
input_61
dense_300_1966855
dense_300_1966857
dense_301_1966882
dense_301_1966884
dense_302_1966909
dense_302_1966911
dense_303_1966936
dense_303_1966938
dense_304_1966963
dense_304_1966965
identity??!dense_300/StatefulPartitionedCall?!dense_301/StatefulPartitionedCall?!dense_302/StatefulPartitionedCall?!dense_303/StatefulPartitionedCall?!dense_304/StatefulPartitionedCall?
!dense_300/StatefulPartitionedCallStatefulPartitionedCallinput_61dense_300_1966855dense_300_1966857*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_300_layer_call_and_return_conditional_losses_19668442#
!dense_300/StatefulPartitionedCall?
!dense_301/StatefulPartitionedCallStatefulPartitionedCall*dense_300/StatefulPartitionedCall:output:0dense_301_1966882dense_301_1966884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_301_layer_call_and_return_conditional_losses_19668712#
!dense_301/StatefulPartitionedCall?
!dense_302/StatefulPartitionedCallStatefulPartitionedCall*dense_301/StatefulPartitionedCall:output:0dense_302_1966909dense_302_1966911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_302_layer_call_and_return_conditional_losses_19668982#
!dense_302/StatefulPartitionedCall?
!dense_303/StatefulPartitionedCallStatefulPartitionedCall*dense_302/StatefulPartitionedCall:output:0dense_303_1966936dense_303_1966938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_303_layer_call_and_return_conditional_losses_19669252#
!dense_303/StatefulPartitionedCall?
!dense_304/StatefulPartitionedCallStatefulPartitionedCall*dense_303/StatefulPartitionedCall:output:0dense_304_1966963dense_304_1966965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_304_layer_call_and_return_conditional_losses_19669522#
!dense_304/StatefulPartitionedCall?
IdentityIdentity*dense_304/StatefulPartitionedCall:output:0"^dense_300/StatefulPartitionedCall"^dense_301/StatefulPartitionedCall"^dense_302/StatefulPartitionedCall"^dense_303/StatefulPartitionedCall"^dense_304/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_300/StatefulPartitionedCall!dense_300/StatefulPartitionedCall2F
!dense_301/StatefulPartitionedCall!dense_301/StatefulPartitionedCall2F
!dense_302/StatefulPartitionedCall!dense_302/StatefulPartitionedCall2F
!dense_303/StatefulPartitionedCall!dense_303/StatefulPartitionedCall2F
!dense_304/StatefulPartitionedCall!dense_304/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_61
?
?
E__inference_model_60_layer_call_and_return_conditional_losses_1967030

inputs
dense_300_1967004
dense_300_1967006
dense_301_1967009
dense_301_1967011
dense_302_1967014
dense_302_1967016
dense_303_1967019
dense_303_1967021
dense_304_1967024
dense_304_1967026
identity??!dense_300/StatefulPartitionedCall?!dense_301/StatefulPartitionedCall?!dense_302/StatefulPartitionedCall?!dense_303/StatefulPartitionedCall?!dense_304/StatefulPartitionedCall?
!dense_300/StatefulPartitionedCallStatefulPartitionedCallinputsdense_300_1967004dense_300_1967006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_300_layer_call_and_return_conditional_losses_19668442#
!dense_300/StatefulPartitionedCall?
!dense_301/StatefulPartitionedCallStatefulPartitionedCall*dense_300/StatefulPartitionedCall:output:0dense_301_1967009dense_301_1967011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_301_layer_call_and_return_conditional_losses_19668712#
!dense_301/StatefulPartitionedCall?
!dense_302/StatefulPartitionedCallStatefulPartitionedCall*dense_301/StatefulPartitionedCall:output:0dense_302_1967014dense_302_1967016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_302_layer_call_and_return_conditional_losses_19668982#
!dense_302/StatefulPartitionedCall?
!dense_303/StatefulPartitionedCallStatefulPartitionedCall*dense_302/StatefulPartitionedCall:output:0dense_303_1967019dense_303_1967021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_303_layer_call_and_return_conditional_losses_19669252#
!dense_303/StatefulPartitionedCall?
!dense_304/StatefulPartitionedCallStatefulPartitionedCall*dense_303/StatefulPartitionedCall:output:0dense_304_1967024dense_304_1967026*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_304_layer_call_and_return_conditional_losses_19669522#
!dense_304/StatefulPartitionedCall?
IdentityIdentity*dense_304/StatefulPartitionedCall:output:0"^dense_300/StatefulPartitionedCall"^dense_301/StatefulPartitionedCall"^dense_302/StatefulPartitionedCall"^dense_303/StatefulPartitionedCall"^dense_304/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_300/StatefulPartitionedCall!dense_300/StatefulPartitionedCall2F
!dense_301/StatefulPartitionedCall!dense_301/StatefulPartitionedCall2F
!dense_302/StatefulPartitionedCall!dense_302/StatefulPartitionedCall2F
!dense_303/StatefulPartitionedCall!dense_303/StatefulPartitionedCall2F
!dense_304/StatefulPartitionedCall!dense_304/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_301_layer_call_and_return_conditional_losses_1967301

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_dense_304_layer_call_fn_1967370

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_304_layer_call_and_return_conditional_losses_19669522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_dense_303_layer_call_fn_1967350

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_303_layer_call_and_return_conditional_losses_19669252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
F__inference_dense_300_layer_call_and_return_conditional_losses_1966844

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?:
?
"__inference__wrapped_model_1966829
input_615
1model_60_dense_300_matmul_readvariableop_resource6
2model_60_dense_300_biasadd_readvariableop_resource5
1model_60_dense_301_matmul_readvariableop_resource6
2model_60_dense_301_biasadd_readvariableop_resource5
1model_60_dense_302_matmul_readvariableop_resource6
2model_60_dense_302_biasadd_readvariableop_resource5
1model_60_dense_303_matmul_readvariableop_resource6
2model_60_dense_303_biasadd_readvariableop_resource5
1model_60_dense_304_matmul_readvariableop_resource6
2model_60_dense_304_biasadd_readvariableop_resource
identity??)model_60/dense_300/BiasAdd/ReadVariableOp?(model_60/dense_300/MatMul/ReadVariableOp?)model_60/dense_301/BiasAdd/ReadVariableOp?(model_60/dense_301/MatMul/ReadVariableOp?)model_60/dense_302/BiasAdd/ReadVariableOp?(model_60/dense_302/MatMul/ReadVariableOp?)model_60/dense_303/BiasAdd/ReadVariableOp?(model_60/dense_303/MatMul/ReadVariableOp?)model_60/dense_304/BiasAdd/ReadVariableOp?(model_60/dense_304/MatMul/ReadVariableOp?
(model_60/dense_300/MatMul/ReadVariableOpReadVariableOp1model_60_dense_300_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(model_60/dense_300/MatMul/ReadVariableOp?
model_60/dense_300/MatMulMatMulinput_610model_60/dense_300/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_300/MatMul?
)model_60/dense_300/BiasAdd/ReadVariableOpReadVariableOp2model_60_dense_300_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_60/dense_300/BiasAdd/ReadVariableOp?
model_60/dense_300/BiasAddBiasAdd#model_60/dense_300/MatMul:product:01model_60/dense_300/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_300/BiasAdd?
model_60/dense_300/SigmoidSigmoid#model_60/dense_300/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_300/Sigmoid?
(model_60/dense_301/MatMul/ReadVariableOpReadVariableOp1model_60_dense_301_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_60/dense_301/MatMul/ReadVariableOp?
model_60/dense_301/MatMulMatMulmodel_60/dense_300/Sigmoid:y:00model_60/dense_301/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_301/MatMul?
)model_60/dense_301/BiasAdd/ReadVariableOpReadVariableOp2model_60_dense_301_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_60/dense_301/BiasAdd/ReadVariableOp?
model_60/dense_301/BiasAddBiasAdd#model_60/dense_301/MatMul:product:01model_60/dense_301/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_301/BiasAdd?
model_60/dense_301/SigmoidSigmoid#model_60/dense_301/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_301/Sigmoid?
(model_60/dense_302/MatMul/ReadVariableOpReadVariableOp1model_60_dense_302_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_60/dense_302/MatMul/ReadVariableOp?
model_60/dense_302/MatMulMatMulmodel_60/dense_301/Sigmoid:y:00model_60/dense_302/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_302/MatMul?
)model_60/dense_302/BiasAdd/ReadVariableOpReadVariableOp2model_60_dense_302_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_60/dense_302/BiasAdd/ReadVariableOp?
model_60/dense_302/BiasAddBiasAdd#model_60/dense_302/MatMul:product:01model_60/dense_302/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_302/BiasAdd?
model_60/dense_302/SigmoidSigmoid#model_60/dense_302/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_302/Sigmoid?
(model_60/dense_303/MatMul/ReadVariableOpReadVariableOp1model_60_dense_303_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_60/dense_303/MatMul/ReadVariableOp?
model_60/dense_303/MatMulMatMulmodel_60/dense_302/Sigmoid:y:00model_60/dense_303/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_303/MatMul?
)model_60/dense_303/BiasAdd/ReadVariableOpReadVariableOp2model_60_dense_303_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_60/dense_303/BiasAdd/ReadVariableOp?
model_60/dense_303/BiasAddBiasAdd#model_60/dense_303/MatMul:product:01model_60/dense_303/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_303/BiasAdd?
model_60/dense_303/SigmoidSigmoid#model_60/dense_303/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_60/dense_303/Sigmoid?
(model_60/dense_304/MatMul/ReadVariableOpReadVariableOp1model_60_dense_304_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(model_60/dense_304/MatMul/ReadVariableOp?
model_60/dense_304/MatMulMatMulmodel_60/dense_303/Sigmoid:y:00model_60/dense_304/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_60/dense_304/MatMul?
)model_60/dense_304/BiasAdd/ReadVariableOpReadVariableOp2model_60_dense_304_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_60/dense_304/BiasAdd/ReadVariableOp?
model_60/dense_304/BiasAddBiasAdd#model_60/dense_304/MatMul:product:01model_60/dense_304/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_60/dense_304/BiasAdd?
model_60/dense_304/SigmoidSigmoid#model_60/dense_304/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_60/dense_304/Sigmoid?
IdentityIdentitymodel_60/dense_304/Sigmoid:y:0*^model_60/dense_300/BiasAdd/ReadVariableOp)^model_60/dense_300/MatMul/ReadVariableOp*^model_60/dense_301/BiasAdd/ReadVariableOp)^model_60/dense_301/MatMul/ReadVariableOp*^model_60/dense_302/BiasAdd/ReadVariableOp)^model_60/dense_302/MatMul/ReadVariableOp*^model_60/dense_303/BiasAdd/ReadVariableOp)^model_60/dense_303/MatMul/ReadVariableOp*^model_60/dense_304/BiasAdd/ReadVariableOp)^model_60/dense_304/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2V
)model_60/dense_300/BiasAdd/ReadVariableOp)model_60/dense_300/BiasAdd/ReadVariableOp2T
(model_60/dense_300/MatMul/ReadVariableOp(model_60/dense_300/MatMul/ReadVariableOp2V
)model_60/dense_301/BiasAdd/ReadVariableOp)model_60/dense_301/BiasAdd/ReadVariableOp2T
(model_60/dense_301/MatMul/ReadVariableOp(model_60/dense_301/MatMul/ReadVariableOp2V
)model_60/dense_302/BiasAdd/ReadVariableOp)model_60/dense_302/BiasAdd/ReadVariableOp2T
(model_60/dense_302/MatMul/ReadVariableOp(model_60/dense_302/MatMul/ReadVariableOp2V
)model_60/dense_303/BiasAdd/ReadVariableOp)model_60/dense_303/BiasAdd/ReadVariableOp2T
(model_60/dense_303/MatMul/ReadVariableOp(model_60/dense_303/MatMul/ReadVariableOp2V
)model_60/dense_304/BiasAdd/ReadVariableOp)model_60/dense_304/BiasAdd/ReadVariableOp2T
(model_60/dense_304/MatMul/ReadVariableOp(model_60/dense_304/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_61
?	
?
F__inference_dense_300_layer_call_and_return_conditional_losses_1967281

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?1
?
E__inference_model_60_layer_call_and_return_conditional_losses_1967220

inputs,
(dense_300_matmul_readvariableop_resource-
)dense_300_biasadd_readvariableop_resource,
(dense_301_matmul_readvariableop_resource-
)dense_301_biasadd_readvariableop_resource,
(dense_302_matmul_readvariableop_resource-
)dense_302_biasadd_readvariableop_resource,
(dense_303_matmul_readvariableop_resource-
)dense_303_biasadd_readvariableop_resource,
(dense_304_matmul_readvariableop_resource-
)dense_304_biasadd_readvariableop_resource
identity?? dense_300/BiasAdd/ReadVariableOp?dense_300/MatMul/ReadVariableOp? dense_301/BiasAdd/ReadVariableOp?dense_301/MatMul/ReadVariableOp? dense_302/BiasAdd/ReadVariableOp?dense_302/MatMul/ReadVariableOp? dense_303/BiasAdd/ReadVariableOp?dense_303/MatMul/ReadVariableOp? dense_304/BiasAdd/ReadVariableOp?dense_304/MatMul/ReadVariableOp?
dense_300/MatMul/ReadVariableOpReadVariableOp(dense_300_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02!
dense_300/MatMul/ReadVariableOp?
dense_300/MatMulMatMulinputs'dense_300/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_300/MatMul?
 dense_300/BiasAdd/ReadVariableOpReadVariableOp)dense_300_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_300/BiasAdd/ReadVariableOp?
dense_300/BiasAddBiasAdddense_300/MatMul:product:0(dense_300/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_300/BiasAdd
dense_300/SigmoidSigmoiddense_300/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_300/Sigmoid?
dense_301/MatMul/ReadVariableOpReadVariableOp(dense_301_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_301/MatMul/ReadVariableOp?
dense_301/MatMulMatMuldense_300/Sigmoid:y:0'dense_301/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_301/MatMul?
 dense_301/BiasAdd/ReadVariableOpReadVariableOp)dense_301_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_301/BiasAdd/ReadVariableOp?
dense_301/BiasAddBiasAdddense_301/MatMul:product:0(dense_301/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_301/BiasAdd
dense_301/SigmoidSigmoiddense_301/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_301/Sigmoid?
dense_302/MatMul/ReadVariableOpReadVariableOp(dense_302_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_302/MatMul/ReadVariableOp?
dense_302/MatMulMatMuldense_301/Sigmoid:y:0'dense_302/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_302/MatMul?
 dense_302/BiasAdd/ReadVariableOpReadVariableOp)dense_302_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_302/BiasAdd/ReadVariableOp?
dense_302/BiasAddBiasAdddense_302/MatMul:product:0(dense_302/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_302/BiasAdd
dense_302/SigmoidSigmoiddense_302/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_302/Sigmoid?
dense_303/MatMul/ReadVariableOpReadVariableOp(dense_303_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_303/MatMul/ReadVariableOp?
dense_303/MatMulMatMuldense_302/Sigmoid:y:0'dense_303/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_303/MatMul?
 dense_303/BiasAdd/ReadVariableOpReadVariableOp)dense_303_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_303/BiasAdd/ReadVariableOp?
dense_303/BiasAddBiasAdddense_303/MatMul:product:0(dense_303/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_303/BiasAdd
dense_303/SigmoidSigmoiddense_303/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_303/Sigmoid?
dense_304/MatMul/ReadVariableOpReadVariableOp(dense_304_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_304/MatMul/ReadVariableOp?
dense_304/MatMulMatMuldense_303/Sigmoid:y:0'dense_304/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_304/MatMul?
 dense_304/BiasAdd/ReadVariableOpReadVariableOp)dense_304_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_304/BiasAdd/ReadVariableOp?
dense_304/BiasAddBiasAdddense_304/MatMul:product:0(dense_304/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_304/BiasAdd
dense_304/SigmoidSigmoiddense_304/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_304/Sigmoid?
IdentityIdentitydense_304/Sigmoid:y:0!^dense_300/BiasAdd/ReadVariableOp ^dense_300/MatMul/ReadVariableOp!^dense_301/BiasAdd/ReadVariableOp ^dense_301/MatMul/ReadVariableOp!^dense_302/BiasAdd/ReadVariableOp ^dense_302/MatMul/ReadVariableOp!^dense_303/BiasAdd/ReadVariableOp ^dense_303/MatMul/ReadVariableOp!^dense_304/BiasAdd/ReadVariableOp ^dense_304/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2D
 dense_300/BiasAdd/ReadVariableOp dense_300/BiasAdd/ReadVariableOp2B
dense_300/MatMul/ReadVariableOpdense_300/MatMul/ReadVariableOp2D
 dense_301/BiasAdd/ReadVariableOp dense_301/BiasAdd/ReadVariableOp2B
dense_301/MatMul/ReadVariableOpdense_301/MatMul/ReadVariableOp2D
 dense_302/BiasAdd/ReadVariableOp dense_302/BiasAdd/ReadVariableOp2B
dense_302/MatMul/ReadVariableOpdense_302/MatMul/ReadVariableOp2D
 dense_303/BiasAdd/ReadVariableOp dense_303/BiasAdd/ReadVariableOp2B
dense_303/MatMul/ReadVariableOpdense_303/MatMul/ReadVariableOp2D
 dense_304/BiasAdd/ReadVariableOp dense_304/BiasAdd/ReadVariableOp2B
dense_304/MatMul/ReadVariableOpdense_304/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
>
input_612
serving_default_input_61:0??????????=
	dense_3040
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?9
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
*s&call_and_return_all_conditional_losses
t__call__
u_default_save_signature"?6
_tf_keras_network?5{"class_name": "Functional", "name": "model_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_61"}, "name": "input_61", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_300", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_300", "inbound_nodes": [[["input_61", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_301", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_301", "inbound_nodes": [[["dense_300", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_302", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_302", "inbound_nodes": [[["dense_301", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_303", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_303", "inbound_nodes": [[["dense_302", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_304", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_304", "inbound_nodes": [[["dense_303", 0, 0, {}]]]}], "input_layers": [["input_61", 0, 0]], "output_layers": [["dense_304", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_61"}, "name": "input_61", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_300", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_300", "inbound_nodes": [[["input_61", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_301", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_301", "inbound_nodes": [[["dense_300", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_302", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_302", "inbound_nodes": [[["dense_301", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_303", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_303", "inbound_nodes": [[["dense_302", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_304", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_304", "inbound_nodes": [[["dense_303", 0, 0, {}]]]}], "input_layers": [["input_61", 0, 0]], "output_layers": [["dense_304", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "binary_accuracy"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_61", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_61"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_300", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_300", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_301", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_301", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_302", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_302", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
*|&call_and_return_all_conditional_losses
}__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_303", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_303", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
*~&call_and_return_all_conditional_losses
__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_304", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_304", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
+iter

,beta_1

-beta_2
	.decay
/learning_ratem_m`mambmcmdme mf%mg&mhvivjvkvlvmvnvo vp%vq&vr"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
 7
%8
&9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
 7
%8
&9"
trackable_list_wrapper
?
0layer_regularization_losses
1non_trainable_variables
regularization_losses

2layers
3layer_metrics
	trainable_variables
4metrics

	variables
t__call__
u_default_save_signature
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!	?@2dense_300/kernel
:@2dense_300/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5layer_regularization_losses
6non_trainable_variables

7layers
8layer_metrics
trainable_variables
	variables
9metrics
regularization_losses
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
": @@2dense_301/kernel
:@2dense_301/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:layer_regularization_losses
;non_trainable_variables

<layers
=layer_metrics
trainable_variables
	variables
>metrics
regularization_losses
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
": @@2dense_302/kernel
:@2dense_302/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_regularization_losses
@non_trainable_variables

Alayers
Blayer_metrics
trainable_variables
	variables
Cmetrics
regularization_losses
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
": @@2dense_303/kernel
:@2dense_303/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_regularization_losses
Enon_trainable_variables

Flayers
Glayer_metrics
!trainable_variables
"	variables
Hmetrics
#regularization_losses
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
": @2dense_304/kernel
:2dense_304/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ilayer_regularization_losses
Jnon_trainable_variables

Klayers
Llayer_metrics
'trainable_variables
(	variables
Mmetrics
)regularization_losses
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
5
N0
O1
P2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	Qtotal
	Rcount
S	variables
T	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Utotal
	Vcount
W
_fn_kwargs
X	variables
Y	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "binary_accuracy"}}
?
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
Q0
R1"
trackable_list_wrapper
-
S	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
(:&	?@2Adam/dense_300/kernel/m
!:@2Adam/dense_300/bias/m
':%@@2Adam/dense_301/kernel/m
!:@2Adam/dense_301/bias/m
':%@@2Adam/dense_302/kernel/m
!:@2Adam/dense_302/bias/m
':%@@2Adam/dense_303/kernel/m
!:@2Adam/dense_303/bias/m
':%@2Adam/dense_304/kernel/m
!:2Adam/dense_304/bias/m
(:&	?@2Adam/dense_300/kernel/v
!:@2Adam/dense_300/bias/v
':%@@2Adam/dense_301/kernel/v
!:@2Adam/dense_301/bias/v
':%@@2Adam/dense_302/kernel/v
!:@2Adam/dense_302/bias/v
':%@@2Adam/dense_303/kernel/v
!:@2Adam/dense_303/bias/v
':%@2Adam/dense_304/kernel/v
!:2Adam/dense_304/bias/v
?2?
E__inference_model_60_layer_call_and_return_conditional_losses_1966998
E__inference_model_60_layer_call_and_return_conditional_losses_1967181
E__inference_model_60_layer_call_and_return_conditional_losses_1967220
E__inference_model_60_layer_call_and_return_conditional_losses_1966969?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_model_60_layer_call_fn_1967107
*__inference_model_60_layer_call_fn_1967270
*__inference_model_60_layer_call_fn_1967053
*__inference_model_60_layer_call_fn_1967245?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_1966829?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *(?%
#? 
input_61??????????
?2?
F__inference_dense_300_layer_call_and_return_conditional_losses_1967281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_300_layer_call_fn_1967290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_301_layer_call_and_return_conditional_losses_1967301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_301_layer_call_fn_1967310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_302_layer_call_and_return_conditional_losses_1967321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_302_layer_call_fn_1967330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_303_layer_call_and_return_conditional_losses_1967341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_303_layer_call_fn_1967350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_304_layer_call_and_return_conditional_losses_1967361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_304_layer_call_fn_1967370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1967142input_61"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1966829w
 %&2?/
(?%
#? 
input_61??????????
? "5?2
0
	dense_304#? 
	dense_304??????????
F__inference_dense_300_layer_call_and_return_conditional_losses_1967281]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? 
+__inference_dense_300_layer_call_fn_1967290P0?-
&?#
!?
inputs??????????
? "??????????@?
F__inference_dense_301_layer_call_and_return_conditional_losses_1967301\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_301_layer_call_fn_1967310O/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_dense_302_layer_call_and_return_conditional_losses_1967321\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_302_layer_call_fn_1967330O/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_dense_303_layer_call_and_return_conditional_losses_1967341\ /?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_303_layer_call_fn_1967350O /?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_dense_304_layer_call_and_return_conditional_losses_1967361\%&/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ~
+__inference_dense_304_layer_call_fn_1967370O%&/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_model_60_layer_call_and_return_conditional_losses_1966969o
 %&:?7
0?-
#? 
input_61??????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_60_layer_call_and_return_conditional_losses_1966998o
 %&:?7
0?-
#? 
input_61??????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_60_layer_call_and_return_conditional_losses_1967181m
 %&8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_60_layer_call_and_return_conditional_losses_1967220m
 %&8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
*__inference_model_60_layer_call_fn_1967053b
 %&:?7
0?-
#? 
input_61??????????
p

 
? "???????????
*__inference_model_60_layer_call_fn_1967107b
 %&:?7
0?-
#? 
input_61??????????
p 

 
? "???????????
*__inference_model_60_layer_call_fn_1967245`
 %&8?5
.?+
!?
inputs??????????
p

 
? "???????????
*__inference_model_60_layer_call_fn_1967270`
 %&8?5
.?+
!?
inputs??????????
p 

 
? "???????????
%__inference_signature_wrapper_1967142?
 %&>?;
? 
4?1
/
input_61#? 
input_61??????????"5?2
0
	dense_304#? 
	dense_304?????????