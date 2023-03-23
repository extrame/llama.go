package ml

import (
	"fmt"
	"os"
	// "atomic"
	// "fmt"
)

const (
	MAX_DIMS     = 4
	MAX_NODES    = 4096
	MAX_PARAMS   = 16
	MAX_CONTEXTS = 64
	MAX_OPT      = 4

	QK = 32 // quantization
)

type dtype uint8

// TODO FP8, BFLOAT16
const (
	TYPE_Q4_0 dtype = iota
	TYPE_Q4_1
	TYPE_I8
	TYPE_I16
	TYPE_I32
	TYPE_F16   // TODO FP16
	TYPE_F32   // TODO FP32
	TYPE_COUNT // NB! COUNT should be the last
)

var BLCK_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{QK, QK, 1, 1, 1, 1, 1}

var TYPE_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{ /* 4 + QK/2 */ 1 /* 4*2 + QK/2 */, 1, 1, 2, 4, 2, 4} // FIXME

func TypeSizeFloat(dt dtype) float32 {
	return float32(TYPE_SIZE[dt]) / float32(BLCK_SIZE[dt]) // FIXME
}

// available tensor operations
type optype uint8

const (
	OP_NONE optype = iota
	OP_DUP
	OP_ADD
	OP_SUB
	OP_MUL
	OP_DIV
	OP_SQR
	OP_SQRT
	OP_SUM
	OP_MEAN
	OP_REPEAT
	OP_ABS
	OP_SGN
	OP_NEG
	OP_STEP
	OP_RELU
	OP_GELU
	OP_SILU
	OP_NORM // normalize
	OP_RMS_NORM

	OP_MUL_MAT

	OP_SCALE
	OP_CPY
	OP_RESHAPE
	OP_VIEW
	OP_PERMUTE
	OP_TRANSPOSE
	OP_GET_ROWS
	OP_DIAG_MASK_INF
	OP_SOFT_MAX
	OP_ROPE
	OP_CONV_1D_1S
	OP_CONV_1D_2S

	OP_FLASH_ATTN
	OP_FLASH_FF

	OP_COUNT
)

// n-dimensional tensor
type Tensor struct {
	Type dtype

	Dims uint32
	NE   [MAX_DIMS]uint32 // number of elements
	//NB   [MAX_DIMS]uint32 // stride in bytes:
	// nb[0] = sizeof(type)
	// nb[1] = nb[0]   * ne[0] + padding
	// nb[i] = nb[i-1] * ne[i-1]

	// compute data
	op optype

	isParam bool

	grad *Tensor
	src0 *Tensor
	src1 *Tensor
	opt  [MAX_OPT]*Tensor

	// thread scheduling
	TasksCount uint32

	// performance
	//perfRuns   uint32
	//perfCycles uint32
	//perfTime   uint64

	Data []float32
	//padding [8]byte
}

func AreSameShape(a, b *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return (a.NE[0] == b.NE[0]) && (a.NE[1] == b.NE[1]) && (a.NE[2] == b.NE[2]) && (a.NE[3] == b.NE[3])
}

func (t *Tensor) Nelements() uint32 {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return t.NE[0] * t.NE[1] * t.NE[2] * t.NE[3]
}

func Nrows(tensor *Tensor) uint32 {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return tensor.NE[1] * tensor.NE[2] * tensor.NE[3]
}

// struct ggml_tensor * ggml_view_tensor(
func ViewTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], src.Data)
}

// ggml.c : ggml_dup_tensor
func DupTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], nil)
}

// struct ggml_tensor * Mul(
func Mul(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, false)
}

// struct ggml_tensor * Mul_inplace(
func MulInplace(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, true)
}

// struct ggml_tensor * Mul_impl(
func MulImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	if !AreSameShape(a, b) {
		fmt.Printf("\n[STOP] MulImpl - tensors of different shapes!")
		os.Exit(1)
	}

	isNode := false

	if inplace && (a.grad != nil || b.grad != nil) {
		isNode = true
	}

	if inplace {
		////ASSERT(is_node == false);
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_MUL
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
func CanMulMat(t0, t1 *Tensor) bool {

	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");

	return (t0.NE[0] == t1.NE[0]) && (t0.NE[2] == t1.NE[2]) && (t0.NE[3] == t1.NE[3]) // FIXME Where NE[1] ??
}

// Mul_mat

// struct ggml_tensor * Mul_mat(
func MulMat(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_can_mul_mat(a, b));

	isNode := false

	if a.grad != nil || b.grad != nil {
		isNode = true
	}

	////const int ne[4] = { a.ne[1], b.ne[1], a.ne[2], b.ne[3] };
	result := NewTensor(ctx, TYPE_F32, min(a.Dims, b.Dims), a.NE[1], b.NE[1], a.NE[2], b.NE[3], nil) // Check for indexes

	result.op = OP_MUL_MAT
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_add

func AddImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	//bool is_node = false;

	////if (!inplace && (a.grad || b.grad)) {
	////is_node = true;
	////}

	////struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_ADD
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Add(ctx *Context, a, b *Tensor) *Tensor {
	return AddImpl(ctx, a, b, false)
}

func AddInplace(ctx *Context, a, b *Tensor) *Tensor {
	return AddImpl(ctx, a, b, true)
}

// ggml_sum

func Sum(ctx *Context, a *Tensor) *Tensor {
	isNode := false

	if a.grad != nil {
		isNode = true
	}

	result := NewTensor1D(ctx, a.Type, 1)

	result.op = OP_SUM
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_sub

func SubImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	////bool is_node = false;

	////if (!inplace && (a.grad || b.grad)) {
	////is_node = true;
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SUB
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Sub(ctx *Context, a, b *Tensor) *Tensor {
	return SubImpl(ctx, a, b, false)
}

func SubInplace(ctx *Context, a, b *Tensor) *Tensor {
	return SubImpl(ctx, a, b, true)
}

// ggml_div

func DivImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	////bool is_node = false;

	////if (!inplace && (a->grad || b->grad)) {
	////is_node = true;
	////}

	////if (inplace) {
	////ASSERT(is_node == false);
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_DIV
	////result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Div(ctx *Context, a, b *Tensor) *Tensor {
	return DivImpl(ctx, a, b, false)
}

func DivInplace(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	return DivImpl(ctx, a, b, true)
}

// ggml_sgn

func SgnImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		isNode = true
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SGN
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Sgn(ctx *Context, a *Tensor) *Tensor {
	return SgnImpl(ctx, a, false)
}

func SgnInplace(ctx *Context, a *Tensor) *Tensor {
	return SgnImpl(ctx, a, true)
}

// Repeat

// struct ggml_tensor * Repeat(
func Repeat(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_can_repeat(a, b));

	isNode := false

	if a.grad != nil {
		isNode = true
	}

	if AreSameShape(a, b) && !isNode {
		return a
	}

	//struct ggml_tensor * result = ggml_new_tensor(ctx, a.type, b.n_dims, b.ne);
	result := NewTensor(ctx, a.Type, b.Dims, b.NE[0], b.NE[1], b.NE[2], b.NE[3], nil)

	result.op = OP_REPEAT
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func IsScalar(tensor *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return tensor.NE[0] == 1 && tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsVector(tensor *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsMatrix(tensor *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return tensor.NE[2] == 1 && tensor.NE[3] == 1
}

// ggml_get_rows

func GetRows(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b.type == TYPE_I32);
	if !IsMatrix(a) || !IsVector(b) /* FIXME || b.Type != TYPE_I32 */ {
		fmt.Printf("\n[ERROR] GetRows fail basic assertions")
		os.Exit(1)
	}

	isNode := false

	if a.grad != nil || b.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows") // FIXME ??
		os.Exit(1)                        // FIXME ??
	}

	// TODO: implement non F32 return
	//struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a.type, a.ne[0], b.ne[0]);
	result := NewTensor2D(ctx, TYPE_F32, a.NE[0], b.NE[0])

	result.op = OP_GET_ROWS
	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	result.src0 = a
	result.src1 = b

	return result
}

func RMSNorm(ctx *Context, a *Tensor) *Tensor {
	return RMSNormImpl(ctx, a, false)
}

func RMSNormInplace(ctx *Context, a *Tensor) *Tensor {
	return RMSNormImpl(ctx, a, true)
}

// //struct ggml_tensor * ggml_rms_norm_impl(
func RMSNormImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows") // FIXME ??
		os.Exit(1)                        // FIXME ??
	}

	////struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_RMS_NORM
	result.src0 = a
	result.src1 = nil // TODO: maybe store epsilon here?

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_view_1d

func View1D(ctx *Context, a *Tensor, ne0 uint32 /*, offset uint64*/) *Tensor {
	if a.grad != nil {
		////ASSERT(false); // gradient propagation is not supported
		fmt.Printf("\n[STOP] View1D : gradient propagation is not supported")
		os.Exit(1)
	}

	result := NewTensor(ctx, a.Type, 1, ne0, 1, 1, 1, a.Data /*+ offset*/) // FIXME

	result.op = OP_VIEW
	result.grad = nil
	result.src0 = a
	result.src1 = nil // TODO: maybe store the offset here?

	return result
}

// static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand) {
func BuildForwardImpl(graph *Graph, tensor *Tensor, expand bool) {

	if !expand {
		graph.NodesCount = 0
		graph.LeafsCount = 0
	}

	n0 := graph.NodesCount
	////UNUSED(n0); // FIXED

	VisitParents(graph, tensor)

	n_new := graph.NodesCount - n0
	////PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

	if n_new > 0 {
		// the last added node should always be starting point
		////ASSERT(cgraph.nodes[cgraph.n_nodes - 1] == tensor);
		if !(graph.Nodes[graph.NodesCount-1] == tensor) {
			fmt.Printf("\n[STOP] BuildForwardImpl : the last added node should always be starting point!")
			os.Exit(1)
		}
	}
}

// void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
func BuildForwardExpand(graph *Graph, tensor *Tensor) {
	BuildForwardImpl(graph, tensor, true)
}

// static void ggml_visit_parents(struct ggml_cgraph * cgraph, struct ggml_tensor * node) {
func VisitParents(graph *Graph, node *Tensor) {

	if node.grad == nil {
		// this usually happens when we generate intermediate nodes from constants in the backward pass
		// it can also happen during forward pass, if the user performs computations with constants
		if node.op != OP_NONE {
			//PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node.op);
		}
	}

	// check if already visited
	for i := uint32(0); i < graph.NodesCount; i++ {
		if graph.Nodes[i] == node {
			return
		}
	}

	for i := uint32(0); i < graph.LeafsCount; i++ {
		if graph.Leafs[i] == node {
			return
		}
	}

	if node.src0 != nil {
		VisitParents(graph, node.src0)
	}

	if node.src1 != nil {
		VisitParents(graph, node.src1)
	}

	for i := 0; i < MAX_OPT; i++ {
		if node.opt[i] != nil {
			VisitParents(graph, node.opt[i])
		}
	}

	if node.op == OP_NONE && node.grad == nil {
		// reached a leaf node, not part of the gradient graph (e.g. a constant)
		////ASSERT(cgraph.n_leafs < MAX_NODES);

		graph.Leafs[graph.LeafsCount] = node
		graph.LeafsCount++
	} else {
		////ASSERT(cgraph.n_nodes < MAX_NODES);

		graph.Nodes[graph.NodesCount] = node
		graph.Grads[graph.NodesCount] = node.grad
		graph.NodesCount++
	}
}

// ggml_cpy

func CopyImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_nelements(a) == ggml_nelements(b));

	isNode := false

	if !inplace && (a.grad != nil || b.grad != nil) {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] cpyImpl")
		os.Exit(1)
	}

	// make a view of the destination
	result := ViewTensor(ctx, b)

	result.op = OP_CPY
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Copy(ctx *Context, a, b *Tensor) *Tensor {
	return CopyImpl(ctx, a, b, false)
}

func CopyInplace(ctx *Context, a, b *Tensor) *Tensor {
	return CopyImpl(ctx, a, b, true)
}

// computation graph
type Graph struct {
	NodesCount uint32 // FIXME Do not need
	LeafsCount uint32 // FIXME Do not need
	Threads    uint32

	WorkSize uint32
	Work     *Tensor
	/*
		Nodes [MAX_NODES]*Tensor
		Grads [MAX_NODES]*Tensor
		Leafs [MAX_NODES]*Tensor
	*/
	Nodes [MAX_NODES]*Tensor
	Grads [MAX_NODES]*Tensor
	Leafs [MAX_NODES]*Tensor

	// performance
	//perfRuns   uint64
	//perfCycles uint64
	////int64_t perf_time_us;
}

type State struct {
	Contexts [MAX_CONTEXTS]ContextContainer
}

type ContextContainer struct {
	Used bool
	Ctx  Context
}

// global state
var gState State
var gStateBarrier int // FIXME atomic_int

type InitParams struct {
	// memory pool
	MemSize   uint64 // bytes
	MemBuffer []byte // if NULL, memory will be allocated internally
}

// scratch buffer
type Scratch struct {
	Offs uint64
	Size uint64
	Data []byte
}

type Object struct {
	Offs uint64
	Size uint64

	Next *Object

	//Padding [8]byte
}

// ml/ggml.c:2248
type Context struct {
	//MemSize        uint64
	//MemBuffer      []byte
	//MemBufferOwned bool

	//Objects uint64
	//Objects []Object // FIXME Speedup with *Object?

	//ObjectsBegin *Object
	//ObjectsEnd   *Object

	//Scratch     Scratch
	//ScratchSave Scratch
}

/*
// ggml_new_tensor
func NewTensor(ctx *Context, dt dtype, dims, ne0, ne1, ne2, ne3 uint32) *Tensor {
	return NewTensorImpl(ctx, dt, dims, ne0, ne1, ne2, ne3, nil)
}
*/
// ggml_new_tensor_1d
func NewTensor1D(ctx *Context, dt dtype, ne uint32) *Tensor {
	return NewTensor(ctx, dt, 1, ne, 1, 1, 1, nil)
}

// ggml_new_tensor_2d
func NewTensor2D(ctx *Context, dt dtype, ne0, ne1 uint32) *Tensor {
	//ne := []uint32{ne0, ne1}
	//return NewTensor(ctx, typ, 2, ne)
	return NewTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil) // FIXME
}

func NewTensor3D(ctx *Context, dt dtype, ne0, ne1, ne2 uint32) *Tensor {
	return NewTensor(ctx, dt, 3, ne0, ne1, ne2, 1, nil) // FIXME
}

func NewTensor4D(ctx *Context, dt dtype, ne0, ne1, ne2, ne3 uint32) *Tensor {
	return NewTensor(ctx, dt, 4, ne0, ne1, ne2, ne3, nil) // FIXME
}

// TODO ne2 for 3D tensors?
// ggml_new_tensor_impl
// func NewTensorImpl(ctx *Context, dt dtype, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {
func NewTensor(ctx *Context, dt dtype, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	if dt != TYPE_F32 && dt != TYPE_I32 {
		fmt.Printf("\n[ERROR] NewTensorImpl got not supported type : %d", dt)
		os.Exit(1)
	}

	// always insert objects at the end of the context's memory pool
	////struct ggml_object * obj_cur = ctx.objects_end;

	////const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur.offs;
	////const size_t cur_size = obj_cur == NULL ? 0 : obj_cur.size;
	////const size_t cur_end  = cur_offs + cur_size;

	//sizeNeeded := uint64(0)

	//if data == nil {
	////size_needed += TYPE_SIZE[type]*(ne[0]/BLCK_SIZE[type]);
	////for (int i = 1; i < n_dims; i++) {
	////    size_needed *= ne[i];
	////}
	// align to MEM_ALIGN
	////size_needed = ((size_needed + MEM_ALIGN - 1)/MEM_ALIGN)*MEM_ALIGN;
	//}

	////char * const mem_buffer = ctx.mem_buffer;
	////struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

	//if ctx.Scratch.Data == nil || data != nil {
	////size_needed += sizeof(struct ggml_tensor);

	////if (cur_end + size_needed + OBJECT_SIZE > ctx.mem_size) {
	////PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
	////    __func__, cur_end + size_needed + OBJECT_SIZE, ctx.mem_size);
	////assert(false);
	////return NULL;
	////}

	////objNew := &Object{
	//Offs: cur_end + OBJECT_SIZE,
	////Size: 0, // FIXME size_needed,
	////Next: nil,
	////}

	//} else {

	//	if ctx.Scratch.Offs+sizeNeeded > ctx.Scratch.Size {
	//PRINT("%s: not enough space in the scratch memory\n", __func__);
	//assert(false);
	//		return nil
	//	}
	//}

	////if (cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE > ctx.mem_size) {
	////PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
	////    __func__, cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE, ctx.mem_size);
	////assert(false);
	////return NULL;
	////}

	////data = (char * const) ctx.scratch.data + ctx.scratch.offs;

	////*obj_new = (struct ggml_object) {
	////.offs = cur_end + OBJECT_SIZE,
	////.size = sizeof(struct ggml_tensor),
	////.next = NULL,
	////};

	//printf("scratch offs = %zu, size_needed = %zu\n", ctx.scratch.offs, size_needed);

	////ctx.scratch.offs += size_needed;
	////}

	//if objCur != nil {
	//	objCur.Next = objNew
	//} else {
	// this is the first object in this context
	//	ctx.ObjectsBegin = objNew
	//}

	//ctx.ObjectsEnd = objNew

	//printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new.size);

	////struct ggml_tensor * const result = (struct ggml_tensor *)(mem_buffer + obj_new.offs);

	////ggml_assert_aligned(result);

	var retData []float32
	if data == nil {
		retData = make([]float32, ne0*ne1*ne2*ne3)
	} else {
		retData = data
	}

	return &Tensor{
		Type: dt,
		Dims: dims,
		NE:   [4]uint32{ne0, ne1, ne2, ne3},
		//NB:   [4]uint32{0, 0, 0, 0},
		op:   OP_NONE,
		opt:  [4]*Tensor{nil, nil, nil, nil},
		Data: retData,
	}
}

// ggml_permute

func Permute(ctx *Context, a *Tensor, axis0, axis1, axis2, axis3 uint32) *Tensor {

	////ASSERT(axis0 >= 0 && axis0 < MAX_DIMS);
	////ASSERT(axis1 >= 0 && axis1 < MAX_DIMS);
	////ASSERT(axis2 >= 0 && axis2 < MAX_DIMS);
	////ASSERT(axis3 >= 0 && axis3 < MAX_DIMS);

	////ASSERT(axis0 != axis1);
	////ASSERT(axis0 != axis2);
	////ASSERT(axis0 != axis3);
	////ASSERT(axis1 != axis2);
	////ASSERT(axis1 != axis3);
	////ASSERT(axis2 != axis3);

	isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] Permute error")
		os.Exit(1)
	}

	result := ViewTensor(ctx, a)

	var ne [MAX_DIMS]uint32
	////int nb[MAX_DIMS];

	ne[axis0] = a.NE[0]
	ne[axis1] = a.NE[1]
	ne[axis2] = a.NE[2]
	ne[axis3] = a.NE[3]

	////nb[axis0] = a.NB[0]
	////nb[axis1] = a.NB[1]
	////nb[axis2] = a.NB[2]
	////nb[axis3] = a.NB[3]

	result.NE[0] = ne[0]
	result.NE[1] = ne[1]
	result.NE[2] = ne[2]
	result.NE[3] = ne[3]

	////result.nb[0] = nb[0];
	////result.nb[1] = nb[1];
	////result.nb[2] = nb[2];
	////result.nb[3] = nb[3];

	result.op = OP_PERMUTE
	result.src0 = a
	result.src1 = nil // TODO: maybe store the permutation here?

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_rope

func Rope(ctx *Context, a *Tensor, past, dims, mode uint32) *Tensor {
	////ASSERT(n_past >= 0);

	isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] Rope error")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	b := NewTensor1D(ctx, TYPE_I32, 3)
	////((int32_t *) b.data)[0] = past
	b.Data[0] = float32(past)
	////((int32_t *) b.data)[1] = dims
	b.Data[1] = float32(dims)
	////((int32_t *) b.data)[2] = mode
	b.Data[2] = float32(mode)

	result.op = OP_ROPE
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Reshape3D(ctx *Context, a *Tensor, ne0, ne1, ne2 uint32) *Tensor {
	////ASSERT(ggml_is_contiguous(a));
	////ASSERT(ggml_nelements(a) == ne0*ne1*ne2);
	if a.Nelements() != ne0*ne1*ne2 {
		fmt.Printf("\n[STOP] Reshape3D : different elements number")
		os.Exit(1)
	}

	////bool is_node = false;

	////if (a.grad) {
	////   //// ASSERT(false); // TODO: implement backward
	////    is_node = true;
	////}

	//ne := [3]uint32{ ne0, ne1, ne2 }
	result := NewTensor(ctx, a.Type, 3, ne0, ne1, ne2, 1, a.Data)

	result.op = OP_RESHAPE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

// struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
func NewFP32(ctx *Context, value float32) *Tensor {

	////ctx.scratch_save = ctx.scratch;
	////ctx.scratch.data = NULL;

	result := NewTensor1D(ctx, TYPE_F32, 1)

	////ctx.scratch = ctx.scratch_save;

	SetFP32(result, value)

	return result
}

// struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
func SetFP32(tensor *Tensor, value float32) *Tensor {

	////n := tensor.Nrows()
	////nc := tensor.NE[0]
	////n1 := tensor.nb[1];

	////data := tensor.Data

	////switch (tensor.type) {
	////case TYPE_Q4_0:
	////{
	////ASSERT(false);
	////} break;
	////case TYPE_Q4_1:
	////{
	////ASSERT(false);
	////} break;
	////case TYPE_I8:
	////{
	////assert(tensor.nb[0] == sizeof(int8_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_I16:
	////{
	////assert(tensor.nb[0] == sizeof(int16_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_I32:
	////{
	////assert(tensor.nb[0] == sizeof(int32_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_F16:
	////{
	////assert(tensor.nb[0] == sizeof(ggml_fp16_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_F32:
	////{
	////assert(tensor.nb[0] == sizeof(float));

	// FIXME Optimize with mem zeroing
	n := tensor.Nelements()
	for i := uint32(0); i < n; i++ {
		////ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
		tensor.Data[i] = value
	}

	////} break;
	////case TYPE_COUNT:
	////{
	////ASSERT(false);
	////} break;
	////}

	return tensor
}

// ggml_scale

func ScaleImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_is_scalar(b));
	////ASSERT(ggml_is_padded_1d(a));

	////bool is_node = false;

	if !inplace && (a.grad != nil || b.grad != nil) {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] ScaleImpl : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	result.op = OP_SCALE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Scale(ctx *Context, a, b *Tensor) *Tensor {
	return ScaleImpl(ctx, a, b, false)
}

func ScaleInplace(ctx *Context, a, b *Tensor) *Tensor {
	return ScaleImpl(ctx, a, b, true)
}

// ggml_diag_mask_inf

func DiagMaskInf(ctx *Context, a *Tensor, past uint32) *Tensor {
	////bool is_node = false;

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] DiagMaskInf : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)
	//// FIXME
	//// b := NewI32(ctx, past)
	b := NewFP32(ctx, float32(past))

	result.op = OP_DIAG_MASK_INF
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

// ggml_soft_max

func SoftMax(ctx *Context, a *Tensor) *Tensor {
	////bool is_node = false;

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] SoftMax : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	result.op = OP_SOFT_MAX
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

// ggml_silu

func SiluImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	////bool is_node = false;

	////if (!inplace && (a.grad)) {
	////is_node = true;
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SILU
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

func Silu(ctx *Context, a *Tensor) *Tensor {
	return SiluImpl(ctx, a, false)
}

func SiluInplace(ctx *Context, a *Tensor) *Tensor {
	return SiluImpl(ctx, a, true)
}

// ggml_step

func StepImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		isNode = true
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_STEP
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Step(ctx *Context, a *Tensor) *Tensor {
	return StepImpl(ctx, a, false)
}

func StepInplace(ctx *Context, a *Tensor) *Tensor {
	return StepImpl(ctx, a, true)
}

// ggml_transpose

func Transpose(ctx *Context, a *Tensor) *Tensor {
	////isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
	}

	result := ViewTensor(ctx, a)

	result.NE[0] = a.NE[1]
	result.NE[1] = a.NE[0]

	//result->nb[0] = a->nb[1];
	//result->nb[1] = a->nb[0];

	result.op = OP_TRANSPOSE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

/*
func BuildForwardImpl(graph *Graph, tensor *Tensor, expand bool) {

	if !expand {
		graph.NodesCount = 0
		graph.LeafsCount = 0
	}

	n0 := graph.NodesCount
	////UNUSED(n0); FIXME ASAP

	VisitParents(graph, tensor)

	newCount := graph.NodesCount - n0
	////PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

	if newCount > 0 {
		// the last added node should always be starting point
		////ASSERT(cgraph.nodes[cgraph.n_nodes - 1] == tensor);
	}
}

func BuildForwardExpand(graph *Graph, tensor *Tensor) {
	BuildForwardImpl(graph, tensor, true)
}*/

func BuildForward(tensor *Tensor) *Graph {

	result := Graph{
		NodesCount: 0,
		LeafsCount: 0,
		// .threads    = 0,
		// .work_size    = 0,
		// *.work         = NULL,

		// FIXME Do use [4096] or [] with append?
		//Nodes: make([4096]*Tensor, 0),
		//Grads: nil,
		//Leafs: nil,

		//.perf_runs    = 0,
		//.perf_cycles  = 0,
		//.perf_time_us = 0,
	}

	BuildForwardImpl(&result, tensor, false)

	return &result
}

func BuildBackward(ctx *Context, gf *Graph, keep bool) Graph {
	////result = *gf
	result := *gf

	////ASSERT(gf.n_nodes > 0);

	// if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
	if keep {
		for i := uint32(0); i < gf.NodesCount; i++ {
			node := gf.Nodes[i]

			if node.grad != nil {
				node.grad = DupTensor(ctx, node)
				gf.Grads[i] = node.grad
			}
		}
	}

	for i := gf.NodesCount - 1; i >= 0; i-- {
		node := gf.Nodes[i]

		// because we detached the grad nodes from the original graph, we can afford inplace operations
		if node.grad != nil {
			ComputeBackward(ctx, node, keep)
		}
	}

	for i := gf.NodesCount - 1; i >= 0; i-- {
		node := gf.Nodes[i]

		if node.isParam {
			////PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
			BuildForwardImpl(&result, node.grad, true)
		}
	}

	return result
}

////////////////////////////////////////////////////////////////////////////////

func ComputeBackward(ctx *Context, tensor *Tensor, inplace bool) {

	src0 := tensor.src0
	src1 := tensor.src1

	switch tensor.op {

	case OP_DUP:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
	case OP_ADD:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
		if src1.grad != nil {
			src1.grad = AddImpl(ctx, src1.grad, tensor.grad, inplace)
		}
	case OP_SUB:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
		if src1.grad != nil {
			src1.grad = SubImpl(ctx, src1.grad, tensor.grad, inplace)
		}
	case OP_MUL:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx, src1, tensor.grad),
					inplace)
		}
		if src1.grad != nil {
			src1.grad =
				AddImpl(ctx,
					src1.grad,
					Mul(ctx, src0, tensor.grad),
					inplace)
		}
	case OP_DIV:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Div(ctx, tensor.grad, src1),
					inplace)
		}
		if src1.grad != nil {
			src1.grad =
				SubImpl(ctx,
					src1.grad,
					Mul(ctx,
						tensor.grad,
						Div(ctx, tensor, src1)),
					inplace)
		}
	case OP_SQR:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx,
						Mul(ctx, src0, tensor.grad),
						Repeat(ctx, NewFP32(ctx, 2.0), src0)),
					inplace)
		}
	case OP_SQRT:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Div(ctx,
						Repeat(ctx, NewFP32(ctx, 0.5), tensor),
						tensor),
					inplace)
		}
	case OP_SUM:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Repeat(ctx, tensor.grad, src0.grad),
					inplace)
		}
	case OP_MEAN:
		//// ASSERT(false); // TODO: implement
	case OP_REPEAT:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Sum(ctx, tensor.grad),
					inplace)
		}
	case OP_ABS:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx,
						Sgn(ctx, src0),
						tensor.grad),
					inplace)
		}
	case OP_SGN:
		if src0.grad != nil {
			// noop
		}
	case OP_NEG:
		if src0.grad != nil {
			src0.grad = SubImpl(ctx, src0.grad, tensor.grad, inplace)
		}
	case OP_STEP:
		if src0.grad != nil {
			// noop
		}
	case OP_RELU:
		if src0.grad != nil {
			src0.grad = SubImpl(ctx,
				src0.grad,
				Mul(ctx,
					Step(ctx, src0),
					tensor.grad),
				inplace)
		}
	case OP_GELU:
		//// ASSERT(false); // TODO: not implemented
	case OP_SILU:
		//// ASSERT(false); // TODO: not implemented
	case OP_NORM:
		//// ASSERT(false); // TODO: not implemented
	case OP_RMS_NORM:
		//// ASSERT(false); // TODO: not implemented
	case OP_MUL_MAT:
		if src0.grad != nil {
			// TODO: this requires outer product - ggml_out_prod(ctx, src1, tensor.grad);
			//// ASSERT(false);
		}
		if src1.grad != nil {
			src1.grad =
				AddImpl(ctx,
					src1.grad,
					// TODO: fix transpose, the node will break the graph connections
					MulMat(ctx, Transpose(ctx, src0), tensor.grad),
					inplace)
		}
	case OP_SCALE:
		//// ASSERT(false); // TODO: not implemented
	case OP_CPY:
		//// ASSERT(false); // TODO: not implemented
	case OP_RESHAPE:
		//// ASSERT(false); // TODO: not implemented
	case OP_VIEW:
		//// ASSERT(false); // not supported
	case OP_PERMUTE:
		//// ASSERT(false); // TODO: not implemented
	case OP_TRANSPOSE:
		//// ASSERT(false); // TODO: not implemented
	case OP_GET_ROWS:
		//// ASSERT(false); // TODO: not implemented
	case OP_DIAG_MASK_INF:
		//// ASSERT(false); // TODO: not implemented
	case OP_SOFT_MAX:
		//// ASSERT(false); // TODO: not implemented
	case OP_ROPE:
		//// ASSERT(false); // TODO: not implemented
	case OP_CONV_1D_1S:
		//// ASSERT(false); // TODO: not implemented
	case OP_CONV_1D_2S:
		//// ASSERT(false); // TODO: not implemented
	case OP_FLASH_ATTN:
		//// ASSERT(false); // not supported
	case OP_FLASH_FF:
		//// ASSERT(false); // not supported
	case OP_NONE:
		// nop
	case OP_COUNT:
		//// ASSERT(false);
	}
}

// ---

type TaskType uint8

const (
	TASK_INIT     TaskType = 0
	TASK_COMPUTE  TaskType = 1
	TASK_FINALIZE TaskType = 2
)

type ComputeParams struct {
	Type     TaskType
	ith, nth uint32
	// work buffer for all threads
	wsize uint64
	wdata []byte // FIXME *void
}

func GraphCompute(ctx *Context, graph *Graph) {

	threads := graph.Threads
	/*
	   struct ggml_compute_state_shared state_shared = {
	       spin      = LOCK_INITIALIZER,
	       threads = threads,
	       n_ready   = 0,
	       has_work  = false,
	       stop      = false,
	   };
	   struct ggml_compute_state * workers = threads > 1 ? alloca(sizeof(struct ggml_compute_state)*(threads - 1)) : NULL;
	*/
	/*
	   // create thread pool
	   if (threads > 1) {
	       ggml_lock_init(&state_shared.spin);

	       atomic_store(&state_shared.has_work, true);

	       for (int j = 0; j < threads - 1; j++) {
	           workers[j] = (struct ggml_compute_state) {
	               .thrd   = 0,
	               .params = {
	                   .type  = TASK_COMPUTE,
	                   .ith   = j + 1,
	                   .nth   = threads,
	                   .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
	                   .wdata = cgraph->work ? cgraph->work->data : NULL,
	               },
	               .node   = NULL,
	               .shared = &state_shared,
	           };

	           int rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
	           ASSERT(rc == 0);
	           UNUSED(rc);
	       }
	   }
	*/

	// initialize tasks + work buffer
	{
		////workSize := 0

		// thread scheduling for the different operations
		for i := uint32(0); i < graph.NodesCount; i++ {

			node := graph.Nodes[i]

			switch node.op {

			case OP_DUP:
				node.TasksCount = 1
			case OP_ADD:
				////node->n_tasks = threads
				node.TasksCount = threads
			case OP_SUB:
			case OP_MUL:
			case OP_DIV:
			case OP_SQR:
			case OP_SQRT:
			case OP_SUM:
			case OP_MEAN:
			case OP_REPEAT:
			case OP_ABS:
			case OP_SGN:
			case OP_NEG:
			case OP_STEP:
			case OP_RELU:
				node.TasksCount = 1
			case OP_GELU:
				node.TasksCount = threads
			case OP_SILU:
				node.TasksCount = threads
			case OP_NORM:
			case OP_RMS_NORM:
				node.TasksCount = threads
			case OP_MUL_MAT:
				node.TasksCount = threads

				// TODO: use different scheduling for different matrix sizes
				//const int nr0 = ggml_nrows(node->src0);
				//const int nr1 = ggml_nrows(node->src1);

				//node->n_tasks = MIN(threads, MAX(1, nr0/128));
				//printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks = %d\n", nr0, nr1, nr0*nr1, node->n_tasks);

				////cur := 0

				// TODO: better way to determine if the matrix is transposed
				////if (node.src0.NB[1] < node->src0->nb[0]) {
				////cur = ggml_nbytes(node)*node->n_tasks; // TODO: this can become (n_tasks-1)
				// TODO: overestimated by factor of x2 for FP16
				////} else {
				////if node.src0.Type == TYPE_F16 && node.src1.Type == TYPE_F32) {

				//// #if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
				////if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
				////node->n_tasks = 1; // TODO: this actually is doing nothing
				//       the threads are still spinning
				////cur = TYPE_SIZE[TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
				//printf("src0: ne0 = %d, ne1 = %d, ne = %d\n", node->src0->ne[0], node->src0->ne[1], node->src0->ne[0]*node->src0->ne[1]);
				//printf("src1: ne0 = %d, ne1 = %d, ne = %d\n", node->src1->ne[0], node->src1->ne[1], node->src1->ne[0]*node->src1->ne[1]);
				//printf("cur = %zu\n", cur);
				////} else {
				////cur = TYPE_SIZE[TYPE_F16]*ggml_nelements(node->src1);
				////}
				////#else
				////cur = TYPE_SIZE[TYPE_F16]*ggml_nelements(node->src1);
				// #endif
				////} else if (node->src0->type == TYPE_F32 &&
				////node->src1->type == TYPE_F32) {
				////if node.src0.Type == TYPE_F32 && node.src1.Type == TYPE_F32 {
				////cur = 0
				////}
				////} else if (node->src0->type == TYPE_Q4_0 &&
				////node->src1->type == TYPE_F32) {
				// #if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
				////if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
				////node->n_tasks = 1;
				////cur = TYPE_SIZE[TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
				////} else {
				////cur = (TYPE_SIZE[TYPE_Q4_0]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_0];
				////}
				////#else
				////cur = (TYPE_SIZE[TYPE_Q4_0]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_0];
				/// #endif
				////} else if (node->src0->type == TYPE_Q4_1 &&
				////node->src1->type == TYPE_F32) {
				//// #if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
				////if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
				////node->n_tasks = 1;
				////cur = TYPE_SIZE[TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
				////} else {
				////cur = (TYPE_SIZE[TYPE_Q4_1]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_1];
				////}
				////#else
				////cur = (TYPE_SIZE[TYPE_Q4_1]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_1];
				//// #endif
				if node.src0.Type == TYPE_F32 && node.src1.Type == TYPE_F32 {
					////cur = 0
				} else {
					////ASSERT(false);
					fmt.Printf("\n[HALT] Mismatch of data within compute graph!")
					os.Exit(1)
				}

				////work_size = MAX(work_size, cur);

			case OP_SCALE:
				node.TasksCount = threads
			case OP_CPY:
			case OP_RESHAPE:
			case OP_VIEW:
			case OP_PERMUTE:
			case OP_TRANSPOSE:
			case OP_GET_ROWS:
			case OP_DIAG_MASK_INF:
				node.TasksCount = 1
			case OP_SOFT_MAX:
				node.TasksCount = threads
			case OP_ROPE:
				node.TasksCount = 1
			case OP_CONV_1D_1S:
			case OP_CONV_1D_2S:

				node.TasksCount = threads

				////ASSERT(node->src0->ne[3] == 1);
				////ASSERT(node->src1->ne[2] == 1);
				////ASSERT(node->src1->ne[3] == 1);

				////cur := 0
				////nk = node.src0.NE[0]

				////if node.src0.Type == TYPE_F16 &&
				////node.src1.Type == TYPE_F32) {
				////cur = sizeof(ggml_fp16_t)*(
				////    nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +
				////( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
				////);
				////} else if (node->src0->type == TYPE_F32 &&
				////           node->src1->type == TYPE_F32) {
				if node.src0.Type == TYPE_F32 && node.src1.Type == TYPE_F32 {
					////cur = sizeof(float)*(
					////    nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +
					////( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
					////);
				} else {
					////ASSERT(false);
					fmt.Printf("\n[HALT] Mismatch of data within compute graph!")
					os.Exit(1)
				}

				////work_size = MAX(work_size, cur);

			case OP_FLASH_ATTN:

				node.TasksCount = threads

				////cur := 0

				////ne11 := Up(node.src1.NE[1], SOFT_MAX_UNROLL)

				if node.src1.Type == TYPE_F32 {
					////cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
					////cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
				}

				if node.src1.Type == TYPE_F16 {
					////cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
					////cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
					fmt.Printf("\n[HALT] Mismatch of data within compute graph!")
					os.Exit(1)
				}

				////work_size = MAX(work_size, cur);

			case OP_FLASH_FF:
				/*
				   node.TasksCount = threads

				   cur := 0

				   if (node->src1->type == TYPE_F32) {
				       cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
				       cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
				   }

				   if (node->src1->type == TYPE_F16) {
				       cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
				       cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
				   }

				   ////work_size = MAX(work_size, cur)
				*/
			case OP_NONE:

				node.TasksCount = 1

			case OP_COUNT:
				////ASSERT(false);
			}
		}

		////if (cgraph->work != NULL && work_size > cgraph->work_size) {
		////ASSERT(false); // TODO: better handling
		////}

		////if (work_size > 0 && cgraph->work == NULL) {
		////cgraph->work_size = work_size + CACHE_LINE_SIZE*(threads - 1);

		////PRINT_DEBUG("%s: allocating work buffer for graph (%zu bytes)\n", __func__, cgraph->work_size);
		////cgraph->work = ggml_new_tensor_1d(ctx, TYPE_I8, cgraph->work_size);

		// FIXME
		////graph.Work = NewTensor1D(ctx, TYPE_I8, graph.WorkSize)

		fmt.Printf("\n[COMPUTE] graph.WorkSize = %d", graph.WorkSize)
		////graph.Work = NewTensor1D(ctx, TYPE_F16, graph.WorkSize)
		graph.Work = NewTensor1D(ctx, TYPE_F32, graph.WorkSize)

		////}
	}

	////const int64_t perf_start_cycles  = ggml_perf_cycles();
	////const int64_t perf_start_time_us = ggml_perf_time_us();

	for i := uint32(0); i < graph.NodesCount; i++ {
		////PRINT_DEBUG_5("%s: %d/%d\n", __func__, i, cgraph->n_nodes);

		node := graph.Nodes[i]

		// TODO: this could be used to avoid unnecessary computations, but it needs to be improved
		//if (node->grad == NULL && node->perf_runs > 0) {
		//    continue;
		//}

		////const int64_t perf_node_start_cycles  = ggml_perf_cycles();
		////const int64_t perf_node_start_time_us = ggml_perf_time_us();

		// INIT
		params := ComputeParams{
			Type: TASK_INIT,
			ith:  0,
			nth:  node.TasksCount, // node.Threads, // FIXME ASAP
			////wsize: graph.work ? ggml_nbytes(cgraph->work) : 0,
			////wdata: graph.work ? cgraph->work->data : NULL,
		}

		fmt.Printf("\n[COMPUTE] ComputeForward | TASK_INIT | ...")
		ComputeForward(&params, node)

		// --- COMPUTE

		////if node->n_tasks > 1 {

		////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
		////atomic_store(&state_shared.has_work, false);
		////}

		////while (atomic_load(&state_shared.has_work)) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		// launch thread pool
		////for (int j = 0; j < threads - 1; j++) {
		////workers[j].params = (struct ggml_compute_params) {
		////.type  = TASK_COMPUTE,
		////.ith   = j + 1,
		////.nth   = node->n_tasks,
		////.wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
		////.wdata = cgraph->work ? cgraph->work->data : NULL,
		////};
		////workers[j].node = node;
		////}

		////atomic_fetch_sub(&state_shared.n_ready, 1);

		////while (atomic_load(&state_shared.n_ready) > 0) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		////atomic_store(&state_shared.has_work, true);
		////}

		params.Type = TASK_COMPUTE
		fmt.Printf("\n[COMPUTE] ComputeForward | TASK_COMPUTE | ...")
		ComputeForward(&params, node)

		// wait for thread pool
		////if (node->n_tasks > 1) {
		////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
		////atomic_store(&state_shared.has_work, false);
		////}

		////while (atomic_load(&state_shared.has_work)) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		////atomic_fetch_sub(&state_shared.n_ready, 1);

		////while (atomic_load(&state_shared.n_ready) != 0) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}
		////}

		// --- FINALIZE

		////if (node->n_tasks > 1) {
		////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
		////atomic_store(&state_shared.has_work, false);
		////}

		////while (atomic_load(&state_shared.has_work)) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		// launch thread pool
		////for (int j = 0; j < threads - 1; j++) {
		////workers[j].params = (struct ggml_compute_params) {
		////.type  = TASK_FINALIZE,
		////.ith   = j + 1,
		////.nth   = node->n_tasks,
		////.wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
		////.wdata = cgraph->work ? cgraph->work->data : NULL,
		////};
		////workers[j].node = node;
		////}

		////atomic_fetch_sub(&state_shared.n_ready, 1);

		////while (atomic_load(&state_shared.n_ready) > 0) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		////atomic_store(&state_shared.has_work, true);
		////}

		params.Type = TASK_FINALIZE
		fmt.Printf("\n[COMPUTE] ComputeForward | TASK_FINALIZE | ...")
		ComputeForward(&params, node)

		// wait for thread pool
		////if (node->n_tasks > 1) {
		////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
		////atomic_store(&state_shared.has_work, false);
		////}

		////while (atomic_load(&state_shared.has_work)) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		////atomic_fetch_sub(&state_shared.n_ready, 1);

		////while (atomic_load(&state_shared.n_ready) != 0) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}
		////}

		// performance stats (node)
		////{
		////int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_node_start_cycles;
		////int64_t perf_time_us_cur = ggml_perf_time_us() - perf_node_start_time_us;

		////node->perf_runs++;
		////node->perf_cycles  += perf_cycles_cur;
		////node->perf_time_us += perf_time_us_cur;
		////}
	}

	// join thread pool
	////if (threads > 1) {
	////atomic_store(&state_shared.stop, true);
	////atomic_store(&state_shared.has_work, true);

	////for (int j = 0; j < threads - 1; j++) {
	////int rc = ggml_thread_join(workers[j].thrd, NULL);
	////ASSERT(rc == 0);
	////UNUSED(rc);
	////}

	////ggml_lock_destroy(&state_shared.spin);
	////}

	// performance stats (graph)
	////{
	////int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_start_cycles;
	////int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

	////cgraph->perf_runs++;
	////cgraph->perf_cycles  += perf_cycles_cur;
	////cgraph->perf_time_us += perf_time_us_cur;

	////PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
	////        __func__, cgraph->perf_runs,
	////        (double) perf_cycles_cur      / (double) ggml_cycles_per_ms(),
	////        (double) cgraph->perf_cycles  / (double) ggml_cycles_per_ms() / (double) cgraph->perf_runs,
	////        (double) perf_time_us_cur     / 1000.0,
	////        (double) cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
	////}
}

/////////////////////////////////

func ComputeForward(params *ComputeParams, tensor *Tensor) {

	fmt.Printf("\n[COMPUTE] ComputeForward...")

	////ASSERT(params);

	switch tensor.op {

	case OP_DUP:
		////ggml_compute_forward_dup(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_dup")
		os.Exit(1)
	case OP_ADD:
		////ggml_compute_forward_add(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_add")
		os.Exit(1)
	case OP_SUB:
		////ggml_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sub")
		os.Exit(1)
	case OP_MUL:
		////ggml_compute_forward_mul(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_mul")
		os.Exit(1)
	case OP_DIV:
		////ggml_compute_forward_div(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_div")
		os.Exit(1)
	case OP_SQR:
		////ggml_compute_forward_sqr(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sqr")
		os.Exit(1)
	case OP_SQRT:
		////ggml_compute_forward_sqrt(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sqrt")
		os.Exit(1)
	case OP_SUM:
		////ggml_compute_forward_sum(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sum")
		os.Exit(1)
	case OP_MEAN:
		////ggml_compute_forward_mean(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_mean")
		os.Exit(1)
	case OP_REPEAT:
		////ggml_compute_forward_repeat(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_repeat")
		os.Exit(1)
	case OP_ABS:
		////ggml_compute_forward_abs(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_abs")
		os.Exit(1)
	case OP_SGN:
		////ggml_compute_forward_sgn(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sgn")
		os.Exit(1)
	case OP_NEG:
		////ggml_compute_forward_neg(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_neg")
		os.Exit(1)
	case OP_STEP:
		////ggml_compute_forward_step(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_step")
		os.Exit(1)
	case OP_RELU:
		////ggml_compute_forward_relu(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_relu")
		os.Exit(1)
	case OP_GELU:
		////ggml_compute_forward_gelu(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_gelu")
		os.Exit(1)
	case OP_SILU:
		////ggml_compute_forward_silu(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_silu")
		os.Exit(1)
	case OP_NORM:
		////ggml_compute_forward_norm(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_norm")
		os.Exit(1)
	case OP_RMS_NORM:
		////ggml_compute_forward_rms_norm(params, tensor->src0, tensor);
		//fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_rms_norm")
		//os.Exit(1)
		ComputeForwardRMSNormFP32(params, tensor.src0, tensor)
	case OP_MUL_MAT:
		////ggml_compute_forward_mul_mat(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_mul_mat")
		os.Exit(1)
	case OP_SCALE:
		////ggml_compute_forward_scale(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_scale")
		os.Exit(1)
	case OP_CPY:
		////ggml_compute_forward_cpy(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_cpy")
		os.Exit(1)
	case OP_RESHAPE:
		////ggml_compute_forward_reshape(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_reshape")
		os.Exit(1)
	case OP_VIEW:
		////ggml_compute_forward_view(params, tensor->src0);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_view")
		os.Exit(1)
	case OP_PERMUTE:
		////ggml_compute_forward_permute(params, tensor->src0);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_permute")
		os.Exit(1)
	case OP_TRANSPOSE:
		////ggml_compute_forward_transpose(params, tensor->src0);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_transpose")
		os.Exit(1)
	case OP_GET_ROWS:
		////ggml_compute_forward_get_rows(params, tensor->src0, tensor->src1, tensor);
		//fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_rows")
		//os.Exit(1)
		ComputeForwardGetRows(params, tensor.src0, tensor.src1, tensor)
	case OP_DIAG_MASK_INF:
		////ggml_compute_forward_diag_mask_inf(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_diag_mask_inf")
		os.Exit(1)
	case OP_SOFT_MAX:
		////ggml_compute_forward_soft_max(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_soft_max")
		os.Exit(1)
	case OP_ROPE:
		////ggml_compute_forward_rope(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_rope")
		os.Exit(1)
	case OP_CONV_1D_1S:
		////ggml_compute_forward_conv_1d_1s(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_conv_1d_1s")
		os.Exit(1)
	case OP_CONV_1D_2S:
		////ggml_compute_forward_conv_1d_2s(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_conv_1d_2s")
		os.Exit(1)
	case OP_FLASH_ATTN:
		////int32_t t = ggml_get_i32_1d(tensor->opt[1], 0);
		////ASSERT(t == 0 || t == 1);
		////bool masked = t != 0;
		////ggml_compute_forward_flash_attn(params, tensor->src0, tensor->src1, tensor->opt[0], masked, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_flash_attn")
		os.Exit(1)
	case OP_FLASH_FF:
		////ggml_compute_forward_flash_ff(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2], tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_flash_ff")
		os.Exit(1)
	case OP_NONE:
		// nop
	case OP_COUNT:
		////ASSERT(false);
		////fmt.Printf("\n[HALT] ")
		////os.Exit(1)
	}
}

////////////////////////////////////////////////////////////////////////////////

// ---

////func VecCopyFP32(n uint32, y, x float32) {
////for i := uint32(0); i < n; i++ {
////y[i]  = x[i]
////}
////}

// NB! Only FP32
// ggml_compute_forward_get_rows_f32
func ComputeForwardGetRows(params *ComputeParams, src0, src1, dst *Tensor) {

	fmt.Printf(" [ ComputeForwardGetRows ] ")

	////assert(params->ith == 0);

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	nc := src0.NE[0]
	nr := src1.Nelements()

	////assert( dst->ne[0] == nc);
	////assert( dst->ne[1] == nr);
	////assert(src0->nb[0] == sizeof(float));

	if dst.NE[0] != nc || dst.NE[1] != nr {
		fmt.Printf("[HALT]ComputeForwardGetRows : wrong dimensions!")
		os.Exit(1)
	}

	// FIXME Speed-up
	for row := uint32(0); row < nr; row++ {
		for column := uint32(0); column < nc; column++ {
			dst.Data[row*nr+column] = src0.Data[row*nr+column]
		}

		/////VecCopyFP32(nc,
		////(float *) ((char *)  dst->data + i*dst->nb[1]),
		////(float *) ((char *) src0->data + r*src0->nb[1]));

		/*
		   r := int(src1.data[i])
		   ggml_vec_cpy_f32(nc,
		           (float *) ((char *)  dst->data + i*dst->nb[1]),
		           (float *) ((char *) src0->data + r*src0->nb[1]));
		*/
	}
}

// NB! FP32 Only
// ggml_compute_forward_rms_norm_f32
func ComputeForwardRMSNormFP32(params *ComputeParams, src0, dst *Tensor) {
	////GGML_ASSERT(ggml_are_same_shape(src0, dst));

	fmt.Printf(" [ ComputeForwardRMSNormFP32 ] ")

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		fmt.Printf(" [ return ] ")
		return
	}

	////GGML_ASSERT(src0->nb[0] == sizeof(float));

	ith := params.ith
	nth := params.nth

	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	////const size_t nb01 = src0->nb[1];
	////const size_t nb02 = src0->nb[2];
	////const size_t nb03 = src0->nb[3];

	////const size_t nb1 = dst->nb[1];
	////const size_t nb2 = dst->nb[2];
	////const size_t nb3 = dst->nb[3];

	////eps := 1e-5 // TODO: make this a parameter

	// TODO: optimize
	for i03 := uint32(0); i03 < ne03; i03++ {
		for i02 := uint32(0); i02 < ne02; i02++ {
			for i01 := uint32(ith); i01 < ne01; i01 += nth {
				////var x float = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
				////x := src0.Data[i01+i02+i03] // FIXME ASAP

				mean := 0.0
				for i00 := uint32(0); i00 < ne00; i00++ {
					////mean += x[i00] * x[i00] // FIXME ASAP
				}

				mean /= float64(ne00)

				////var y float = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
				////y := dst.Data[i01+i02+i03]

				////memcpy(y, x, ne00 * sizeof(float));

				// WAS COMMENTED
				// for (int i00 = 0; i00 < ne00; i00++) {
				//     y[i00] = x[i00];
				// }

				////scale := float32(1.0 / math.Sqrt(mean+eps))

				////VecScaleFP32(ne00, y, scale) // FIXME ASAP
			}
		}
	}
}

// inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
func VecScaleFP32(n uint32, y []float32, v float32) {
	////#if defined(GGML_SIMD)
	////const int np = (n & ~(GGML_F32_STEP - 1));

	////GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	////GGML_F32_VEC ay[GGML_F32_ARR];

	////for (int i = 0; i < np; i += GGML_F32_STEP) {
	////for (int j = 0; j < GGML_F32_ARR; j++) {
	////ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
	////ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

	////GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
	////}
	////}

	// leftovers
	////for (int i = np; i < n; ++i) {
	////y[i] *= v;
	////}
	////#else
	// scalar
	for i := uint32(0); i < n; i++ {
		y[i] *= v
	}
	////#endif
}

// ---

// uitils.h
type GPTVocab struct {
	Token2ID map[string]uint32
	ID2Token map[uint32]string
}

func NewVocab() *GPTVocab {
	return &GPTVocab{
		Token2ID: make(map[string]uint32),
		ID2Token: make(map[uint32]string),
	}
}

func min(a, b uint32) uint32 {
	if a <= b {
		return a
	}
	return b
}

// FIXME Would it work with UTF-8? Rewrite for runes
// SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
// std::vector<gpt_vocab::id> llamaTokenize(const gpt_vocab & vocab, const std::string & text, bool bos) {
func Tokenize(vocab *GPTVocab, text string, bos bool) []uint32 {

	// TODO: Calculate this constant from the vocabulary
	MAX_TOKEN_LEN := uint32(18)
	length := uint32(len(text))

	////std::vector<gpt_vocab::id> res;
	res := make([]uint32, 0)
	////std::vector<int> score;
	//var score []uint32
	////std::vector<gpt_vocab::id> prev;
	//var prev []uint32
	////int len = text.length();

	////score.resize(len + 1);
	score := make([]uint32, length+1)
	////prev.resize(len + 1);
	prev := make([]uint32, length+1)

	// Forward pass
	for i := uint32(0); i < length; i++ {
		maxLen := min(length-i, MAX_TOKEN_LEN)
		for subLen := uint32(1); subLen <= maxLen; subLen++ {
			////auto sub = text.substr(i, sub_len);
			sub := text[i : i+subLen]
			////auto token = vocab.token_to_id.find(sub);
			token, ok := vocab.Token2ID[sub] // FIXME if not found?
			//if token != vocab.token2id.end() {
			if ok {
				tokenScore := uint32(len(sub) * len(sub))
				localScore := score[i] + tokenScore
				next := i + subLen
				if score[next] < localScore {
					score[next] = localScore
					////prev[next] = (*token).second
					prev[next] = token
				}
			}
		}
	}

	// Backward pass
	i := len(text)
	for i > 0 {
		////gpt_vocab::id token_id = prev[i];
		tokenID := prev[i]
		if tokenID == 0 {
			// TODO: Return error or something more meaningful
			fmt.Printf("\n[ERROR] Failed to tokenize string!")
			break
		}
		////res.push_back(token_id);
		res = append(res, tokenID)
		////auto token = (*vocab.id_to_token.find(token_id)).second;
		token, _ := vocab.ID2Token[tokenID]
		i -= len(token)
	}

	if bos {
		////res.push_back(1); // TODO: replace with vocab.bos
		res = append(res, 1) // TODO: replace with vocab.bos
	}

	// Pieces are in reverse order so correct that
	////std::reverse(res.begin(), res.end());
	//sort.Reverse(sort.IntSlice(res))

	//fmt.Printf("\n\n=== PREV ===\n\n%+v", prev)
	//fmt.Printf("\n\n=== RES ===\n\n%+v", res)

	reversed := make([]uint32, 0, len(res))
	for n := len(res); n > 0; n-- {
		reversed = append(reversed, res[n-1])
	}

	return reversed
}

func Init(params InitParams) *Context {
	// make this function thread safe
	////ggml_critical_section_start();

	isFirstCall := true // FIXME static ??

	if isFirstCall {
		// initialize GELU, SILU and EXP F32 tables
		////{
		////const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

		////ggml_fp16_t ii;
		////for (int i = 0; i < (1 << 16); ++i) {
		////uint16_t ui = i;
		////memcpy(&ii, &ui, sizeof(ii));
		////const float f = table_f32_f16[i] = COMPUTE_FP16_TO_FP32(ii);
		////table_gelu_f16[i] = FP32_TO_FP16(ggml_gelu_f32(f));
		////table_silu_f16[i] = FP32_TO_FP16(ggml_silu_f32(f));
		////table_exp_f16[i]  = FP32_TO_FP16(exp(f));
		////}

		////const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

		////PRINT_DEBUG("%s: GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
		////}

		// initialize g_state
		{
			////const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

			gState = State{
				Contexts: [MAX_CONTEXTS]ContextContainer{},
			}

			for i := uint32(0); i < MAX_CONTEXTS; i++ {
				gState.Contexts[i].Used = false
			}

			////const uint64_t t_end = ggml_time_us(); UNUSED(t_end);
			//var end uint64 = ggml_time_us(); UNUSED(t_end)

			////PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
		}

		isFirstCall = false
	}

	// find non-used context in g_state
	var ctx *Context

	for i := uint32(0); i < MAX_CONTEXTS; i++ {
		if !gState.Contexts[i].Used {
			gState.Contexts[i].Used = true
			ctx = &gState.Contexts[i].Ctx

			////PRINT_DEBUG("%s: found unused context %d\n", __func__, i)
			break
		}
	}

	if ctx == nil {
		////PRINT_DEBUG("%s: no unused context found\n", __func__);
		////ggml_critical_section_end();
		return nil
	}

	//var buf []byte
	//if params.MemBuffer == nil {
	//	buf = make([]byte, params.MemSize)
	//} else {
	//	buf = params.MemBuffer
	//}

	ctx = &Context{
		//MemSize:        params.MemSize,
		//MemBuffer:      buf,
		//MemBufferOwned: params.MemBuffer != nil,
		//Objects:        0,
		//Objects:      make([]Object, 0),
		//ObjectsBegin: nil,
		//ObjectsEnd:   nil,
		//Scratch:      Scratch{0, 0, nil},
		//ScratchSave:  Scratch{0, 0, nil},
	}

	////ggml_assert_aligned(ctx.mem_buffer);

	////PRINT_DEBUG("%s: context initialized\n", __func__);

	////ggml_critical_section_end();

	return ctx
}
