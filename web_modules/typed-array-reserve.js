/** @module typedArrayReserve */
/**
* @typedef {Int8Array
*   | Uint8Array
*   | Uint8ClampedArray
*   | Int16Array
*   | Uint16Array
*   | Int32Array
*   | Uint32Array
*   | Float32Array
*   | Float64Array
*   | BigInt64Array
*   | BigUint64Array} TypedArray
*/
/**
* Typed array of at least `length` elements: `array` when long enough, a new
* array of the same type otherwise.
*
* @param {TypedArray} array
* @param {number} length
* @param {boolean} [copy=false] Copy the values of `array`.
* @param {number} [factor=1.5] Growth factor, above 1. Higher is faster and
*   uses more memory.
* @returns {TypedArray}
* @alias module:typedArrayReserve
* @see Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to algorithms (3rd ed.), §17.4 Dynamic tables. MIT Press.
*/
function typedArrayReserve(array, length, copy = false, factor = 1.5) {
	if (array.length >= length) return array;
	const reserved = new array.constructor(Math.max(length, Math.ceil(array.length * factor)));
	if (copy) reserved.set(array);
	return reserved;
}

export { typedArrayReserve as default };