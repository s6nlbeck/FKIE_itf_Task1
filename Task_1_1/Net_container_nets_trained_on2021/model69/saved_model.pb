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
dense_340/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*!
shared_namedense_340/kernel
v
$dense_340/kernel/Read/ReadVariableOpReadVariableOpdense_340/kernel*
_output_shapes
:	?@*
dtype0
t
dense_340/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_340/bias
m
"dense_340/bias/Read/ReadVariableOpReadVariableOpdense_340/bias*
_output_shapes
:@*
dtype0
|
dense_341/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_341/kernel
u
$dense_341/kernel/Read/ReadVariableOpReadVariableOpdense_341/kernel*
_output_shapes

:@@*
dtype0
t
dense_341/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_341/bias
m
"dense_341/bias/Read/ReadVariableOpReadVariableOpdense_341/bias*
_output_shapes
:@*
dtype0
|
dense_342/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_342/kernel
u
$dense_342/kernel/Read/ReadVariableOpReadVariableOpdense_342/kernel*
_output_shapes

:@@*
dtype0
t
dense_342/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_342/bias
m
"dense_342/bias/Read/ReadVariableOpReadVariableOpdense_342/bias*
_output_shapes
:@*
dtype0
|
dense_343/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_343/kernel
u
$dense_343/kernel/Read/ReadVariableOpReadVariableOpdense_343/kernel*
_output_shapes

:@@*
dtype0
t
dense_343/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_343/bias
m
"dense_343/bias/Read/ReadVariableOpReadVariableOpdense_343/bias*
_output_shapes
:@*
dtype0
|
dense_344/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_344/kernel
u
$dense_344/kernel/Read/ReadVariableOpReadVariableOpdense_344/kernel*
_output_shapes

:@*
dtype0
t
dense_344/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_344/bias
m
"dense_344/bias/Read/ReadVariableOpReadVariableOpdense_344/bias*
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
Adam/dense_340/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*(
shared_nameAdam/dense_340/kernel/m
?
+Adam/dense_340/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_340/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_340/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_340/bias/m
{
)Adam/dense_340/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_340/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_341/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_341/kernel/m
?
+Adam/dense_341/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_341/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_341/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_341/bias/m
{
)Adam/dense_341/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_341/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_342/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_342/kernel/m
?
+Adam/dense_342/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_342/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_342/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_342/bias/m
{
)Adam/dense_342/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_342/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_343/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_343/kernel/m
?
+Adam/dense_343/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_343/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_343/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_343/bias/m
{
)Adam/dense_343/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_343/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_344/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_344/kernel/m
?
+Adam/dense_344/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_344/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_344/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_344/bias/m
{
)Adam/dense_344/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_344/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_340/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*(
shared_nameAdam/dense_340/kernel/v
?
+Adam/dense_340/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_340/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_340/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_340/bias/v
{
)Adam/dense_340/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_340/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_341/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_341/kernel/v
?
+Adam/dense_341/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_341/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_341/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_341/bias/v
{
)Adam/dense_341/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_341/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_342/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_342/kernel/v
?
+Adam/dense_342/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_342/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_342/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_342/bias/v
{
)Adam/dense_342/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_342/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_343/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_343/kernel/v
?
+Adam/dense_343/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_343/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_343/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_343/bias/v
{
)Adam/dense_343/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_343/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_344/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_344/kernel/v
?
+Adam/dense_344/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_344/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_344/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_344/bias/v
{
)Adam/dense_344/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_344/bias/v*
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
VARIABLE_VALUEdense_340/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_340/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_341/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_341/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_342/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_342/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_343/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_343/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_344/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_344/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_340/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_340/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_341/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_341/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_342/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_342/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_343/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_343/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_344/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_344/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_340/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_340/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_341/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_341/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_342/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_342/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_343/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_343/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_344/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_344/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
serving_default_input_69Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_69dense_340/kerneldense_340/biasdense_341/kerneldense_341/biasdense_342/kerneldense_342/biasdense_343/kerneldense_343/biasdense_344/kerneldense_344/bias*
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
%__inference_signature_wrapper_1975518
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_340/kernel/Read/ReadVariableOp"dense_340/bias/Read/ReadVariableOp$dense_341/kernel/Read/ReadVariableOp"dense_341/bias/Read/ReadVariableOp$dense_342/kernel/Read/ReadVariableOp"dense_342/bias/Read/ReadVariableOp$dense_343/kernel/Read/ReadVariableOp"dense_343/bias/Read/ReadVariableOp$dense_344/kernel/Read/ReadVariableOp"dense_344/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/dense_340/kernel/m/Read/ReadVariableOp)Adam/dense_340/bias/m/Read/ReadVariableOp+Adam/dense_341/kernel/m/Read/ReadVariableOp)Adam/dense_341/bias/m/Read/ReadVariableOp+Adam/dense_342/kernel/m/Read/ReadVariableOp)Adam/dense_342/bias/m/Read/ReadVariableOp+Adam/dense_343/kernel/m/Read/ReadVariableOp)Adam/dense_343/bias/m/Read/ReadVariableOp+Adam/dense_344/kernel/m/Read/ReadVariableOp)Adam/dense_344/bias/m/Read/ReadVariableOp+Adam/dense_340/kernel/v/Read/ReadVariableOp)Adam/dense_340/bias/v/Read/ReadVariableOp+Adam/dense_341/kernel/v/Read/ReadVariableOp)Adam/dense_341/bias/v/Read/ReadVariableOp+Adam/dense_342/kernel/v/Read/ReadVariableOp)Adam/dense_342/bias/v/Read/ReadVariableOp+Adam/dense_343/kernel/v/Read/ReadVariableOp)Adam/dense_343/bias/v/Read/ReadVariableOp+Adam/dense_344/kernel/v/Read/ReadVariableOp)Adam/dense_344/bias/v/Read/ReadVariableOpConst*6
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
 __inference__traced_save_1975892
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_340/kerneldense_340/biasdense_341/kerneldense_341/biasdense_342/kerneldense_342/biasdense_343/kerneldense_343/biasdense_344/kerneldense_344/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/dense_340/kernel/mAdam/dense_340/bias/mAdam/dense_341/kernel/mAdam/dense_341/bias/mAdam/dense_342/kernel/mAdam/dense_342/bias/mAdam/dense_343/kernel/mAdam/dense_343/bias/mAdam/dense_344/kernel/mAdam/dense_344/bias/mAdam/dense_340/kernel/vAdam/dense_340/bias/vAdam/dense_341/kernel/vAdam/dense_341/bias/vAdam/dense_342/kernel/vAdam/dense_342/bias/vAdam/dense_343/kernel/vAdam/dense_343/bias/vAdam/dense_344/kernel/vAdam/dense_344/bias/v*5
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
#__inference__traced_restore_1976025??
?	
?
F__inference_dense_343_layer_call_and_return_conditional_losses_1975301

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
+__inference_dense_341_layer_call_fn_1975686

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
F__inference_dense_341_layer_call_and_return_conditional_losses_19752472
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
?
E__inference_model_68_layer_call_and_return_conditional_losses_1975460

inputs
dense_340_1975434
dense_340_1975436
dense_341_1975439
dense_341_1975441
dense_342_1975444
dense_342_1975446
dense_343_1975449
dense_343_1975451
dense_344_1975454
dense_344_1975456
identity??!dense_340/StatefulPartitionedCall?!dense_341/StatefulPartitionedCall?!dense_342/StatefulPartitionedCall?!dense_343/StatefulPartitionedCall?!dense_344/StatefulPartitionedCall?
!dense_340/StatefulPartitionedCallStatefulPartitionedCallinputsdense_340_1975434dense_340_1975436*
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
F__inference_dense_340_layer_call_and_return_conditional_losses_19752202#
!dense_340/StatefulPartitionedCall?
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1975439dense_341_1975441*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_19752472#
!dense_341/StatefulPartitionedCall?
!dense_342/StatefulPartitionedCallStatefulPartitionedCall*dense_341/StatefulPartitionedCall:output:0dense_342_1975444dense_342_1975446*
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
F__inference_dense_342_layer_call_and_return_conditional_losses_19752742#
!dense_342/StatefulPartitionedCall?
!dense_343/StatefulPartitionedCallStatefulPartitionedCall*dense_342/StatefulPartitionedCall:output:0dense_343_1975449dense_343_1975451*
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
F__inference_dense_343_layer_call_and_return_conditional_losses_19753012#
!dense_343/StatefulPartitionedCall?
!dense_344/StatefulPartitionedCallStatefulPartitionedCall*dense_343/StatefulPartitionedCall:output:0dense_344_1975454dense_344_1975456*
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
F__inference_dense_344_layer_call_and_return_conditional_losses_19753282#
!dense_344/StatefulPartitionedCall?
IdentityIdentity*dense_344/StatefulPartitionedCall:output:0"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall"^dense_342/StatefulPartitionedCall"^dense_343/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall2F
!dense_342/StatefulPartitionedCall!dense_342/StatefulPartitionedCall2F
!dense_343/StatefulPartitionedCall!dense_343/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_340_layer_call_and_return_conditional_losses_1975657

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
E__inference_model_68_layer_call_and_return_conditional_losses_1975596

inputs,
(dense_340_matmul_readvariableop_resource-
)dense_340_biasadd_readvariableop_resource,
(dense_341_matmul_readvariableop_resource-
)dense_341_biasadd_readvariableop_resource,
(dense_342_matmul_readvariableop_resource-
)dense_342_biasadd_readvariableop_resource,
(dense_343_matmul_readvariableop_resource-
)dense_343_biasadd_readvariableop_resource,
(dense_344_matmul_readvariableop_resource-
)dense_344_biasadd_readvariableop_resource
identity?? dense_340/BiasAdd/ReadVariableOp?dense_340/MatMul/ReadVariableOp? dense_341/BiasAdd/ReadVariableOp?dense_341/MatMul/ReadVariableOp? dense_342/BiasAdd/ReadVariableOp?dense_342/MatMul/ReadVariableOp? dense_343/BiasAdd/ReadVariableOp?dense_343/MatMul/ReadVariableOp? dense_344/BiasAdd/ReadVariableOp?dense_344/MatMul/ReadVariableOp?
dense_340/MatMul/ReadVariableOpReadVariableOp(dense_340_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02!
dense_340/MatMul/ReadVariableOp?
dense_340/MatMulMatMulinputs'dense_340/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_340/MatMul?
 dense_340/BiasAdd/ReadVariableOpReadVariableOp)dense_340_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_340/BiasAdd/ReadVariableOp?
dense_340/BiasAddBiasAdddense_340/MatMul:product:0(dense_340/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_340/BiasAdd
dense_340/SigmoidSigmoiddense_340/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_340/Sigmoid?
dense_341/MatMul/ReadVariableOpReadVariableOp(dense_341_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_341/MatMul/ReadVariableOp?
dense_341/MatMulMatMuldense_340/Sigmoid:y:0'dense_341/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_341/MatMul?
 dense_341/BiasAdd/ReadVariableOpReadVariableOp)dense_341_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_341/BiasAdd/ReadVariableOp?
dense_341/BiasAddBiasAdddense_341/MatMul:product:0(dense_341/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_341/BiasAdd
dense_341/SigmoidSigmoiddense_341/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_341/Sigmoid?
dense_342/MatMul/ReadVariableOpReadVariableOp(dense_342_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_342/MatMul/ReadVariableOp?
dense_342/MatMulMatMuldense_341/Sigmoid:y:0'dense_342/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_342/MatMul?
 dense_342/BiasAdd/ReadVariableOpReadVariableOp)dense_342_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_342/BiasAdd/ReadVariableOp?
dense_342/BiasAddBiasAdddense_342/MatMul:product:0(dense_342/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_342/BiasAdd
dense_342/SigmoidSigmoiddense_342/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_342/Sigmoid?
dense_343/MatMul/ReadVariableOpReadVariableOp(dense_343_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_343/MatMul/ReadVariableOp?
dense_343/MatMulMatMuldense_342/Sigmoid:y:0'dense_343/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_343/MatMul?
 dense_343/BiasAdd/ReadVariableOpReadVariableOp)dense_343_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_343/BiasAdd/ReadVariableOp?
dense_343/BiasAddBiasAdddense_343/MatMul:product:0(dense_343/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_343/BiasAdd
dense_343/SigmoidSigmoiddense_343/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_343/Sigmoid?
dense_344/MatMul/ReadVariableOpReadVariableOp(dense_344_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_344/MatMul/ReadVariableOp?
dense_344/MatMulMatMuldense_343/Sigmoid:y:0'dense_344/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_344/MatMul?
 dense_344/BiasAdd/ReadVariableOpReadVariableOp)dense_344_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_344/BiasAdd/ReadVariableOp?
dense_344/BiasAddBiasAdddense_344/MatMul:product:0(dense_344/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_344/BiasAdd
dense_344/SigmoidSigmoiddense_344/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_344/Sigmoid?
IdentityIdentitydense_344/Sigmoid:y:0!^dense_340/BiasAdd/ReadVariableOp ^dense_340/MatMul/ReadVariableOp!^dense_341/BiasAdd/ReadVariableOp ^dense_341/MatMul/ReadVariableOp!^dense_342/BiasAdd/ReadVariableOp ^dense_342/MatMul/ReadVariableOp!^dense_343/BiasAdd/ReadVariableOp ^dense_343/MatMul/ReadVariableOp!^dense_344/BiasAdd/ReadVariableOp ^dense_344/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2D
 dense_340/BiasAdd/ReadVariableOp dense_340/BiasAdd/ReadVariableOp2B
dense_340/MatMul/ReadVariableOpdense_340/MatMul/ReadVariableOp2D
 dense_341/BiasAdd/ReadVariableOp dense_341/BiasAdd/ReadVariableOp2B
dense_341/MatMul/ReadVariableOpdense_341/MatMul/ReadVariableOp2D
 dense_342/BiasAdd/ReadVariableOp dense_342/BiasAdd/ReadVariableOp2B
dense_342/MatMul/ReadVariableOpdense_342/MatMul/ReadVariableOp2D
 dense_343/BiasAdd/ReadVariableOp dense_343/BiasAdd/ReadVariableOp2B
dense_343/MatMul/ReadVariableOpdense_343/MatMul/ReadVariableOp2D
 dense_344/BiasAdd/ReadVariableOp dense_344/BiasAdd/ReadVariableOp2B
dense_344/MatMul/ReadVariableOpdense_344/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_model_68_layer_call_fn_1975646

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
E__inference_model_68_layer_call_and_return_conditional_losses_19754602
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
+__inference_dense_344_layer_call_fn_1975746

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
F__inference_dense_344_layer_call_and_return_conditional_losses_19753282
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
?
?
*__inference_model_68_layer_call_fn_1975429
input_69
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
StatefulPartitionedCallStatefulPartitionedCallinput_69unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
E__inference_model_68_layer_call_and_return_conditional_losses_19754062
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
input_69
?	
?
F__inference_dense_340_layer_call_and_return_conditional_losses_1975220

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
E__inference_model_68_layer_call_and_return_conditional_losses_1975557

inputs,
(dense_340_matmul_readvariableop_resource-
)dense_340_biasadd_readvariableop_resource,
(dense_341_matmul_readvariableop_resource-
)dense_341_biasadd_readvariableop_resource,
(dense_342_matmul_readvariableop_resource-
)dense_342_biasadd_readvariableop_resource,
(dense_343_matmul_readvariableop_resource-
)dense_343_biasadd_readvariableop_resource,
(dense_344_matmul_readvariableop_resource-
)dense_344_biasadd_readvariableop_resource
identity?? dense_340/BiasAdd/ReadVariableOp?dense_340/MatMul/ReadVariableOp? dense_341/BiasAdd/ReadVariableOp?dense_341/MatMul/ReadVariableOp? dense_342/BiasAdd/ReadVariableOp?dense_342/MatMul/ReadVariableOp? dense_343/BiasAdd/ReadVariableOp?dense_343/MatMul/ReadVariableOp? dense_344/BiasAdd/ReadVariableOp?dense_344/MatMul/ReadVariableOp?
dense_340/MatMul/ReadVariableOpReadVariableOp(dense_340_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02!
dense_340/MatMul/ReadVariableOp?
dense_340/MatMulMatMulinputs'dense_340/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_340/MatMul?
 dense_340/BiasAdd/ReadVariableOpReadVariableOp)dense_340_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_340/BiasAdd/ReadVariableOp?
dense_340/BiasAddBiasAdddense_340/MatMul:product:0(dense_340/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_340/BiasAdd
dense_340/SigmoidSigmoiddense_340/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_340/Sigmoid?
dense_341/MatMul/ReadVariableOpReadVariableOp(dense_341_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_341/MatMul/ReadVariableOp?
dense_341/MatMulMatMuldense_340/Sigmoid:y:0'dense_341/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_341/MatMul?
 dense_341/BiasAdd/ReadVariableOpReadVariableOp)dense_341_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_341/BiasAdd/ReadVariableOp?
dense_341/BiasAddBiasAdddense_341/MatMul:product:0(dense_341/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_341/BiasAdd
dense_341/SigmoidSigmoiddense_341/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_341/Sigmoid?
dense_342/MatMul/ReadVariableOpReadVariableOp(dense_342_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_342/MatMul/ReadVariableOp?
dense_342/MatMulMatMuldense_341/Sigmoid:y:0'dense_342/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_342/MatMul?
 dense_342/BiasAdd/ReadVariableOpReadVariableOp)dense_342_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_342/BiasAdd/ReadVariableOp?
dense_342/BiasAddBiasAdddense_342/MatMul:product:0(dense_342/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_342/BiasAdd
dense_342/SigmoidSigmoiddense_342/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_342/Sigmoid?
dense_343/MatMul/ReadVariableOpReadVariableOp(dense_343_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_343/MatMul/ReadVariableOp?
dense_343/MatMulMatMuldense_342/Sigmoid:y:0'dense_343/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_343/MatMul?
 dense_343/BiasAdd/ReadVariableOpReadVariableOp)dense_343_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_343/BiasAdd/ReadVariableOp?
dense_343/BiasAddBiasAdddense_343/MatMul:product:0(dense_343/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_343/BiasAdd
dense_343/SigmoidSigmoiddense_343/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_343/Sigmoid?
dense_344/MatMul/ReadVariableOpReadVariableOp(dense_344_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_344/MatMul/ReadVariableOp?
dense_344/MatMulMatMuldense_343/Sigmoid:y:0'dense_344/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_344/MatMul?
 dense_344/BiasAdd/ReadVariableOpReadVariableOp)dense_344_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_344/BiasAdd/ReadVariableOp?
dense_344/BiasAddBiasAdddense_344/MatMul:product:0(dense_344/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_344/BiasAdd
dense_344/SigmoidSigmoiddense_344/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_344/Sigmoid?
IdentityIdentitydense_344/Sigmoid:y:0!^dense_340/BiasAdd/ReadVariableOp ^dense_340/MatMul/ReadVariableOp!^dense_341/BiasAdd/ReadVariableOp ^dense_341/MatMul/ReadVariableOp!^dense_342/BiasAdd/ReadVariableOp ^dense_342/MatMul/ReadVariableOp!^dense_343/BiasAdd/ReadVariableOp ^dense_343/MatMul/ReadVariableOp!^dense_344/BiasAdd/ReadVariableOp ^dense_344/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2D
 dense_340/BiasAdd/ReadVariableOp dense_340/BiasAdd/ReadVariableOp2B
dense_340/MatMul/ReadVariableOpdense_340/MatMul/ReadVariableOp2D
 dense_341/BiasAdd/ReadVariableOp dense_341/BiasAdd/ReadVariableOp2B
dense_341/MatMul/ReadVariableOpdense_341/MatMul/ReadVariableOp2D
 dense_342/BiasAdd/ReadVariableOp dense_342/BiasAdd/ReadVariableOp2B
dense_342/MatMul/ReadVariableOpdense_342/MatMul/ReadVariableOp2D
 dense_343/BiasAdd/ReadVariableOp dense_343/BiasAdd/ReadVariableOp2B
dense_343/MatMul/ReadVariableOpdense_343/MatMul/ReadVariableOp2D
 dense_344/BiasAdd/ReadVariableOp dense_344/BiasAdd/ReadVariableOp2B
dense_344/MatMul/ReadVariableOpdense_344/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?:
?
"__inference__wrapped_model_1975205
input_695
1model_68_dense_340_matmul_readvariableop_resource6
2model_68_dense_340_biasadd_readvariableop_resource5
1model_68_dense_341_matmul_readvariableop_resource6
2model_68_dense_341_biasadd_readvariableop_resource5
1model_68_dense_342_matmul_readvariableop_resource6
2model_68_dense_342_biasadd_readvariableop_resource5
1model_68_dense_343_matmul_readvariableop_resource6
2model_68_dense_343_biasadd_readvariableop_resource5
1model_68_dense_344_matmul_readvariableop_resource6
2model_68_dense_344_biasadd_readvariableop_resource
identity??)model_68/dense_340/BiasAdd/ReadVariableOp?(model_68/dense_340/MatMul/ReadVariableOp?)model_68/dense_341/BiasAdd/ReadVariableOp?(model_68/dense_341/MatMul/ReadVariableOp?)model_68/dense_342/BiasAdd/ReadVariableOp?(model_68/dense_342/MatMul/ReadVariableOp?)model_68/dense_343/BiasAdd/ReadVariableOp?(model_68/dense_343/MatMul/ReadVariableOp?)model_68/dense_344/BiasAdd/ReadVariableOp?(model_68/dense_344/MatMul/ReadVariableOp?
(model_68/dense_340/MatMul/ReadVariableOpReadVariableOp1model_68_dense_340_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(model_68/dense_340/MatMul/ReadVariableOp?
model_68/dense_340/MatMulMatMulinput_690model_68/dense_340/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_340/MatMul?
)model_68/dense_340/BiasAdd/ReadVariableOpReadVariableOp2model_68_dense_340_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_68/dense_340/BiasAdd/ReadVariableOp?
model_68/dense_340/BiasAddBiasAdd#model_68/dense_340/MatMul:product:01model_68/dense_340/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_340/BiasAdd?
model_68/dense_340/SigmoidSigmoid#model_68/dense_340/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_340/Sigmoid?
(model_68/dense_341/MatMul/ReadVariableOpReadVariableOp1model_68_dense_341_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_68/dense_341/MatMul/ReadVariableOp?
model_68/dense_341/MatMulMatMulmodel_68/dense_340/Sigmoid:y:00model_68/dense_341/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_341/MatMul?
)model_68/dense_341/BiasAdd/ReadVariableOpReadVariableOp2model_68_dense_341_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_68/dense_341/BiasAdd/ReadVariableOp?
model_68/dense_341/BiasAddBiasAdd#model_68/dense_341/MatMul:product:01model_68/dense_341/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_341/BiasAdd?
model_68/dense_341/SigmoidSigmoid#model_68/dense_341/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_341/Sigmoid?
(model_68/dense_342/MatMul/ReadVariableOpReadVariableOp1model_68_dense_342_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_68/dense_342/MatMul/ReadVariableOp?
model_68/dense_342/MatMulMatMulmodel_68/dense_341/Sigmoid:y:00model_68/dense_342/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_342/MatMul?
)model_68/dense_342/BiasAdd/ReadVariableOpReadVariableOp2model_68_dense_342_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_68/dense_342/BiasAdd/ReadVariableOp?
model_68/dense_342/BiasAddBiasAdd#model_68/dense_342/MatMul:product:01model_68/dense_342/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_342/BiasAdd?
model_68/dense_342/SigmoidSigmoid#model_68/dense_342/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_342/Sigmoid?
(model_68/dense_343/MatMul/ReadVariableOpReadVariableOp1model_68_dense_343_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_68/dense_343/MatMul/ReadVariableOp?
model_68/dense_343/MatMulMatMulmodel_68/dense_342/Sigmoid:y:00model_68/dense_343/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_343/MatMul?
)model_68/dense_343/BiasAdd/ReadVariableOpReadVariableOp2model_68_dense_343_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_68/dense_343/BiasAdd/ReadVariableOp?
model_68/dense_343/BiasAddBiasAdd#model_68/dense_343/MatMul:product:01model_68/dense_343/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_343/BiasAdd?
model_68/dense_343/SigmoidSigmoid#model_68/dense_343/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_68/dense_343/Sigmoid?
(model_68/dense_344/MatMul/ReadVariableOpReadVariableOp1model_68_dense_344_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(model_68/dense_344/MatMul/ReadVariableOp?
model_68/dense_344/MatMulMatMulmodel_68/dense_343/Sigmoid:y:00model_68/dense_344/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_68/dense_344/MatMul?
)model_68/dense_344/BiasAdd/ReadVariableOpReadVariableOp2model_68_dense_344_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_68/dense_344/BiasAdd/ReadVariableOp?
model_68/dense_344/BiasAddBiasAdd#model_68/dense_344/MatMul:product:01model_68/dense_344/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_68/dense_344/BiasAdd?
model_68/dense_344/SigmoidSigmoid#model_68/dense_344/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_68/dense_344/Sigmoid?
IdentityIdentitymodel_68/dense_344/Sigmoid:y:0*^model_68/dense_340/BiasAdd/ReadVariableOp)^model_68/dense_340/MatMul/ReadVariableOp*^model_68/dense_341/BiasAdd/ReadVariableOp)^model_68/dense_341/MatMul/ReadVariableOp*^model_68/dense_342/BiasAdd/ReadVariableOp)^model_68/dense_342/MatMul/ReadVariableOp*^model_68/dense_343/BiasAdd/ReadVariableOp)^model_68/dense_343/MatMul/ReadVariableOp*^model_68/dense_344/BiasAdd/ReadVariableOp)^model_68/dense_344/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2V
)model_68/dense_340/BiasAdd/ReadVariableOp)model_68/dense_340/BiasAdd/ReadVariableOp2T
(model_68/dense_340/MatMul/ReadVariableOp(model_68/dense_340/MatMul/ReadVariableOp2V
)model_68/dense_341/BiasAdd/ReadVariableOp)model_68/dense_341/BiasAdd/ReadVariableOp2T
(model_68/dense_341/MatMul/ReadVariableOp(model_68/dense_341/MatMul/ReadVariableOp2V
)model_68/dense_342/BiasAdd/ReadVariableOp)model_68/dense_342/BiasAdd/ReadVariableOp2T
(model_68/dense_342/MatMul/ReadVariableOp(model_68/dense_342/MatMul/ReadVariableOp2V
)model_68/dense_343/BiasAdd/ReadVariableOp)model_68/dense_343/BiasAdd/ReadVariableOp2T
(model_68/dense_343/MatMul/ReadVariableOp(model_68/dense_343/MatMul/ReadVariableOp2V
)model_68/dense_344/BiasAdd/ReadVariableOp)model_68/dense_344/BiasAdd/ReadVariableOp2T
(model_68/dense_344/MatMul/ReadVariableOp(model_68/dense_344/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_69
?
?
+__inference_dense_342_layer_call_fn_1975706

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
F__inference_dense_342_layer_call_and_return_conditional_losses_19752742
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
ͬ
?
#__inference__traced_restore_1976025
file_prefix%
!assignvariableop_dense_340_kernel%
!assignvariableop_1_dense_340_bias'
#assignvariableop_2_dense_341_kernel%
!assignvariableop_3_dense_341_bias'
#assignvariableop_4_dense_342_kernel%
!assignvariableop_5_dense_342_bias'
#assignvariableop_6_dense_343_kernel%
!assignvariableop_7_dense_343_bias'
#assignvariableop_8_dense_344_kernel%
!assignvariableop_9_dense_344_bias!
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
+assignvariableop_21_adam_dense_340_kernel_m-
)assignvariableop_22_adam_dense_340_bias_m/
+assignvariableop_23_adam_dense_341_kernel_m-
)assignvariableop_24_adam_dense_341_bias_m/
+assignvariableop_25_adam_dense_342_kernel_m-
)assignvariableop_26_adam_dense_342_bias_m/
+assignvariableop_27_adam_dense_343_kernel_m-
)assignvariableop_28_adam_dense_343_bias_m/
+assignvariableop_29_adam_dense_344_kernel_m-
)assignvariableop_30_adam_dense_344_bias_m/
+assignvariableop_31_adam_dense_340_kernel_v-
)assignvariableop_32_adam_dense_340_bias_v/
+assignvariableop_33_adam_dense_341_kernel_v-
)assignvariableop_34_adam_dense_341_bias_v/
+assignvariableop_35_adam_dense_342_kernel_v-
)assignvariableop_36_adam_dense_342_bias_v/
+assignvariableop_37_adam_dense_343_kernel_v-
)assignvariableop_38_adam_dense_343_bias_v/
+assignvariableop_39_adam_dense_344_kernel_v-
)assignvariableop_40_adam_dense_344_bias_v
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_340_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_340_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_341_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_341_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_342_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_342_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_343_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_343_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_344_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_344_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_340_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_340_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_341_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_341_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_342_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_342_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_343_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_343_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_344_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_344_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_340_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_340_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_341_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_341_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_342_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_342_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_343_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_343_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_344_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_344_bias_vIdentity_40:output:0"/device:CPU:0*
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
?
?
*__inference_model_68_layer_call_fn_1975483
input_69
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
StatefulPartitionedCallStatefulPartitionedCallinput_69unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
E__inference_model_68_layer_call_and_return_conditional_losses_19754602
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
input_69
?	
?
F__inference_dense_342_layer_call_and_return_conditional_losses_1975274

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
F__inference_dense_344_layer_call_and_return_conditional_losses_1975737

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
?
?
E__inference_model_68_layer_call_and_return_conditional_losses_1975374
input_69
dense_340_1975348
dense_340_1975350
dense_341_1975353
dense_341_1975355
dense_342_1975358
dense_342_1975360
dense_343_1975363
dense_343_1975365
dense_344_1975368
dense_344_1975370
identity??!dense_340/StatefulPartitionedCall?!dense_341/StatefulPartitionedCall?!dense_342/StatefulPartitionedCall?!dense_343/StatefulPartitionedCall?!dense_344/StatefulPartitionedCall?
!dense_340/StatefulPartitionedCallStatefulPartitionedCallinput_69dense_340_1975348dense_340_1975350*
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
F__inference_dense_340_layer_call_and_return_conditional_losses_19752202#
!dense_340/StatefulPartitionedCall?
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1975353dense_341_1975355*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_19752472#
!dense_341/StatefulPartitionedCall?
!dense_342/StatefulPartitionedCallStatefulPartitionedCall*dense_341/StatefulPartitionedCall:output:0dense_342_1975358dense_342_1975360*
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
F__inference_dense_342_layer_call_and_return_conditional_losses_19752742#
!dense_342/StatefulPartitionedCall?
!dense_343/StatefulPartitionedCallStatefulPartitionedCall*dense_342/StatefulPartitionedCall:output:0dense_343_1975363dense_343_1975365*
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
F__inference_dense_343_layer_call_and_return_conditional_losses_19753012#
!dense_343/StatefulPartitionedCall?
!dense_344/StatefulPartitionedCallStatefulPartitionedCall*dense_343/StatefulPartitionedCall:output:0dense_344_1975368dense_344_1975370*
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
F__inference_dense_344_layer_call_and_return_conditional_losses_19753282#
!dense_344/StatefulPartitionedCall?
IdentityIdentity*dense_344/StatefulPartitionedCall:output:0"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall"^dense_342/StatefulPartitionedCall"^dense_343/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall2F
!dense_342/StatefulPartitionedCall!dense_342/StatefulPartitionedCall2F
!dense_343/StatefulPartitionedCall!dense_343/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_69
?	
?
F__inference_dense_341_layer_call_and_return_conditional_losses_1975677

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
E__inference_model_68_layer_call_and_return_conditional_losses_1975406

inputs
dense_340_1975380
dense_340_1975382
dense_341_1975385
dense_341_1975387
dense_342_1975390
dense_342_1975392
dense_343_1975395
dense_343_1975397
dense_344_1975400
dense_344_1975402
identity??!dense_340/StatefulPartitionedCall?!dense_341/StatefulPartitionedCall?!dense_342/StatefulPartitionedCall?!dense_343/StatefulPartitionedCall?!dense_344/StatefulPartitionedCall?
!dense_340/StatefulPartitionedCallStatefulPartitionedCallinputsdense_340_1975380dense_340_1975382*
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
F__inference_dense_340_layer_call_and_return_conditional_losses_19752202#
!dense_340/StatefulPartitionedCall?
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1975385dense_341_1975387*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_19752472#
!dense_341/StatefulPartitionedCall?
!dense_342/StatefulPartitionedCallStatefulPartitionedCall*dense_341/StatefulPartitionedCall:output:0dense_342_1975390dense_342_1975392*
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
F__inference_dense_342_layer_call_and_return_conditional_losses_19752742#
!dense_342/StatefulPartitionedCall?
!dense_343/StatefulPartitionedCallStatefulPartitionedCall*dense_342/StatefulPartitionedCall:output:0dense_343_1975395dense_343_1975397*
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
F__inference_dense_343_layer_call_and_return_conditional_losses_19753012#
!dense_343/StatefulPartitionedCall?
!dense_344/StatefulPartitionedCallStatefulPartitionedCall*dense_343/StatefulPartitionedCall:output:0dense_344_1975400dense_344_1975402*
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
F__inference_dense_344_layer_call_and_return_conditional_losses_19753282#
!dense_344/StatefulPartitionedCall?
IdentityIdentity*dense_344/StatefulPartitionedCall:output:0"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall"^dense_342/StatefulPartitionedCall"^dense_343/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall2F
!dense_342/StatefulPartitionedCall!dense_342/StatefulPartitionedCall2F
!dense_343/StatefulPartitionedCall!dense_343/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_343_layer_call_and_return_conditional_losses_1975717

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
+__inference_dense_343_layer_call_fn_1975726

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
F__inference_dense_343_layer_call_and_return_conditional_losses_19753012
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
?
E__inference_model_68_layer_call_and_return_conditional_losses_1975345
input_69
dense_340_1975231
dense_340_1975233
dense_341_1975258
dense_341_1975260
dense_342_1975285
dense_342_1975287
dense_343_1975312
dense_343_1975314
dense_344_1975339
dense_344_1975341
identity??!dense_340/StatefulPartitionedCall?!dense_341/StatefulPartitionedCall?!dense_342/StatefulPartitionedCall?!dense_343/StatefulPartitionedCall?!dense_344/StatefulPartitionedCall?
!dense_340/StatefulPartitionedCallStatefulPartitionedCallinput_69dense_340_1975231dense_340_1975233*
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
F__inference_dense_340_layer_call_and_return_conditional_losses_19752202#
!dense_340/StatefulPartitionedCall?
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1975258dense_341_1975260*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_19752472#
!dense_341/StatefulPartitionedCall?
!dense_342/StatefulPartitionedCallStatefulPartitionedCall*dense_341/StatefulPartitionedCall:output:0dense_342_1975285dense_342_1975287*
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
F__inference_dense_342_layer_call_and_return_conditional_losses_19752742#
!dense_342/StatefulPartitionedCall?
!dense_343/StatefulPartitionedCallStatefulPartitionedCall*dense_342/StatefulPartitionedCall:output:0dense_343_1975312dense_343_1975314*
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
F__inference_dense_343_layer_call_and_return_conditional_losses_19753012#
!dense_343/StatefulPartitionedCall?
!dense_344/StatefulPartitionedCallStatefulPartitionedCall*dense_343/StatefulPartitionedCall:output:0dense_344_1975339dense_344_1975341*
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
F__inference_dense_344_layer_call_and_return_conditional_losses_19753282#
!dense_344/StatefulPartitionedCall?
IdentityIdentity*dense_344/StatefulPartitionedCall:output:0"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall"^dense_342/StatefulPartitionedCall"^dense_343/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall2F
!dense_342/StatefulPartitionedCall!dense_342/StatefulPartitionedCall2F
!dense_343/StatefulPartitionedCall!dense_343/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_69
?	
?
F__inference_dense_344_layer_call_and_return_conditional_losses_1975328

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
?T
?
 __inference__traced_save_1975892
file_prefix/
+savev2_dense_340_kernel_read_readvariableop-
)savev2_dense_340_bias_read_readvariableop/
+savev2_dense_341_kernel_read_readvariableop-
)savev2_dense_341_bias_read_readvariableop/
+savev2_dense_342_kernel_read_readvariableop-
)savev2_dense_342_bias_read_readvariableop/
+savev2_dense_343_kernel_read_readvariableop-
)savev2_dense_343_bias_read_readvariableop/
+savev2_dense_344_kernel_read_readvariableop-
)savev2_dense_344_bias_read_readvariableop(
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
2savev2_adam_dense_340_kernel_m_read_readvariableop4
0savev2_adam_dense_340_bias_m_read_readvariableop6
2savev2_adam_dense_341_kernel_m_read_readvariableop4
0savev2_adam_dense_341_bias_m_read_readvariableop6
2savev2_adam_dense_342_kernel_m_read_readvariableop4
0savev2_adam_dense_342_bias_m_read_readvariableop6
2savev2_adam_dense_343_kernel_m_read_readvariableop4
0savev2_adam_dense_343_bias_m_read_readvariableop6
2savev2_adam_dense_344_kernel_m_read_readvariableop4
0savev2_adam_dense_344_bias_m_read_readvariableop6
2savev2_adam_dense_340_kernel_v_read_readvariableop4
0savev2_adam_dense_340_bias_v_read_readvariableop6
2savev2_adam_dense_341_kernel_v_read_readvariableop4
0savev2_adam_dense_341_bias_v_read_readvariableop6
2savev2_adam_dense_342_kernel_v_read_readvariableop4
0savev2_adam_dense_342_bias_v_read_readvariableop6
2savev2_adam_dense_343_kernel_v_read_readvariableop4
0savev2_adam_dense_343_bias_v_read_readvariableop6
2savev2_adam_dense_344_kernel_v_read_readvariableop4
0savev2_adam_dense_344_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_340_kernel_read_readvariableop)savev2_dense_340_bias_read_readvariableop+savev2_dense_341_kernel_read_readvariableop)savev2_dense_341_bias_read_readvariableop+savev2_dense_342_kernel_read_readvariableop)savev2_dense_342_bias_read_readvariableop+savev2_dense_343_kernel_read_readvariableop)savev2_dense_343_bias_read_readvariableop+savev2_dense_344_kernel_read_readvariableop)savev2_dense_344_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_dense_340_kernel_m_read_readvariableop0savev2_adam_dense_340_bias_m_read_readvariableop2savev2_adam_dense_341_kernel_m_read_readvariableop0savev2_adam_dense_341_bias_m_read_readvariableop2savev2_adam_dense_342_kernel_m_read_readvariableop0savev2_adam_dense_342_bias_m_read_readvariableop2savev2_adam_dense_343_kernel_m_read_readvariableop0savev2_adam_dense_343_bias_m_read_readvariableop2savev2_adam_dense_344_kernel_m_read_readvariableop0savev2_adam_dense_344_bias_m_read_readvariableop2savev2_adam_dense_340_kernel_v_read_readvariableop0savev2_adam_dense_340_bias_v_read_readvariableop2savev2_adam_dense_341_kernel_v_read_readvariableop0savev2_adam_dense_341_bias_v_read_readvariableop2savev2_adam_dense_342_kernel_v_read_readvariableop0savev2_adam_dense_342_bias_v_read_readvariableop2savev2_adam_dense_343_kernel_v_read_readvariableop0savev2_adam_dense_343_bias_v_read_readvariableop2savev2_adam_dense_344_kernel_v_read_readvariableop0savev2_adam_dense_344_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
*__inference_model_68_layer_call_fn_1975621

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
E__inference_model_68_layer_call_and_return_conditional_losses_19754062
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
?
?
%__inference_signature_wrapper_1975518
input_69
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
StatefulPartitionedCallStatefulPartitionedCallinput_69unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
"__inference__wrapped_model_19752052
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
input_69
?	
?
F__inference_dense_341_layer_call_and_return_conditional_losses_1975247

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
F__inference_dense_342_layer_call_and_return_conditional_losses_1975697

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
+__inference_dense_340_layer_call_fn_1975666

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
F__inference_dense_340_layer_call_and_return_conditional_losses_19752202
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
input_692
serving_default_input_69:0??????????=
	dense_3440
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
_tf_keras_network?5{"class_name": "Functional", "name": "model_68", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_68", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_69"}, "name": "input_69", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_340", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_340", "inbound_nodes": [[["input_69", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_341", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_341", "inbound_nodes": [[["dense_340", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_342", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_342", "inbound_nodes": [[["dense_341", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_343", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_343", "inbound_nodes": [[["dense_342", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_344", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_344", "inbound_nodes": [[["dense_343", 0, 0, {}]]]}], "input_layers": [["input_69", 0, 0]], "output_layers": [["dense_344", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_68", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_69"}, "name": "input_69", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_340", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_340", "inbound_nodes": [[["input_69", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_341", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_341", "inbound_nodes": [[["dense_340", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_342", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_342", "inbound_nodes": [[["dense_341", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_343", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_343", "inbound_nodes": [[["dense_342", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_344", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_344", "inbound_nodes": [[["dense_343", 0, 0, {}]]]}], "input_layers": [["input_69", 0, 0]], "output_layers": [["dense_344", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "binary_accuracy"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_69", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_69"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_340", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_340", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_341", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_341", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_342", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_342", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
*|&call_and_return_all_conditional_losses
}__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_343", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_343", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
*~&call_and_return_all_conditional_losses
__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_344", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_344", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
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
#:!	?@2dense_340/kernel
:@2dense_340/bias
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
": @@2dense_341/kernel
:@2dense_341/bias
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
": @@2dense_342/kernel
:@2dense_342/bias
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
": @@2dense_343/kernel
:@2dense_343/bias
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
": @2dense_344/kernel
:2dense_344/bias
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
(:&	?@2Adam/dense_340/kernel/m
!:@2Adam/dense_340/bias/m
':%@@2Adam/dense_341/kernel/m
!:@2Adam/dense_341/bias/m
':%@@2Adam/dense_342/kernel/m
!:@2Adam/dense_342/bias/m
':%@@2Adam/dense_343/kernel/m
!:@2Adam/dense_343/bias/m
':%@2Adam/dense_344/kernel/m
!:2Adam/dense_344/bias/m
(:&	?@2Adam/dense_340/kernel/v
!:@2Adam/dense_340/bias/v
':%@@2Adam/dense_341/kernel/v
!:@2Adam/dense_341/bias/v
':%@@2Adam/dense_342/kernel/v
!:@2Adam/dense_342/bias/v
':%@@2Adam/dense_343/kernel/v
!:@2Adam/dense_343/bias/v
':%@2Adam/dense_344/kernel/v
!:2Adam/dense_344/bias/v
?2?
E__inference_model_68_layer_call_and_return_conditional_losses_1975596
E__inference_model_68_layer_call_and_return_conditional_losses_1975557
E__inference_model_68_layer_call_and_return_conditional_losses_1975345
E__inference_model_68_layer_call_and_return_conditional_losses_1975374?
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
*__inference_model_68_layer_call_fn_1975646
*__inference_model_68_layer_call_fn_1975621
*__inference_model_68_layer_call_fn_1975429
*__inference_model_68_layer_call_fn_1975483?
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
"__inference__wrapped_model_1975205?
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
input_69??????????
?2?
F__inference_dense_340_layer_call_and_return_conditional_losses_1975657?
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
+__inference_dense_340_layer_call_fn_1975666?
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1975677?
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
+__inference_dense_341_layer_call_fn_1975686?
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
F__inference_dense_342_layer_call_and_return_conditional_losses_1975697?
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
+__inference_dense_342_layer_call_fn_1975706?
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
F__inference_dense_343_layer_call_and_return_conditional_losses_1975717?
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
+__inference_dense_343_layer_call_fn_1975726?
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
F__inference_dense_344_layer_call_and_return_conditional_losses_1975737?
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
+__inference_dense_344_layer_call_fn_1975746?
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
%__inference_signature_wrapper_1975518input_69"?
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
"__inference__wrapped_model_1975205w
 %&2?/
(?%
#? 
input_69??????????
? "5?2
0
	dense_344#? 
	dense_344??????????
F__inference_dense_340_layer_call_and_return_conditional_losses_1975657]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? 
+__inference_dense_340_layer_call_fn_1975666P0?-
&?#
!?
inputs??????????
? "??????????@?
F__inference_dense_341_layer_call_and_return_conditional_losses_1975677\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_341_layer_call_fn_1975686O/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_dense_342_layer_call_and_return_conditional_losses_1975697\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_342_layer_call_fn_1975706O/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_dense_343_layer_call_and_return_conditional_losses_1975717\ /?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_343_layer_call_fn_1975726O /?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_dense_344_layer_call_and_return_conditional_losses_1975737\%&/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ~
+__inference_dense_344_layer_call_fn_1975746O%&/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_model_68_layer_call_and_return_conditional_losses_1975345o
 %&:?7
0?-
#? 
input_69??????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_68_layer_call_and_return_conditional_losses_1975374o
 %&:?7
0?-
#? 
input_69??????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_68_layer_call_and_return_conditional_losses_1975557m
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
E__inference_model_68_layer_call_and_return_conditional_losses_1975596m
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
*__inference_model_68_layer_call_fn_1975429b
 %&:?7
0?-
#? 
input_69??????????
p

 
? "???????????
*__inference_model_68_layer_call_fn_1975483b
 %&:?7
0?-
#? 
input_69??????????
p 

 
? "???????????
*__inference_model_68_layer_call_fn_1975621`
 %&8?5
.?+
!?
inputs??????????
p

 
? "???????????
*__inference_model_68_layer_call_fn_1975646`
 %&8?5
.?+
!?
inputs??????????
p 

 
? "???????????
%__inference_signature_wrapper_1975518?
 %&>?;
? 
4?1
/
input_69#? 
input_69??????????"5?2
0
	dense_344#? 
	dense_344?????????