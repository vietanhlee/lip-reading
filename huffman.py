import heapq
import os
from collections import defaultdict

class HuffmanNode:
    def __init__(self, freq, byte=None, left=None, right=None):
        self.freq = freq
        self.byte = byte
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoding:
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}

    def build_frequency_dict(self, data_bytes):
        freq = defaultdict(int)
        for b in data_bytes:
            freq[b] += 1
        return freq

    def build_heap(self, freq_dict):
        heap = []
        for byte, frequency in freq_dict.items():
            node = HuffmanNode(frequency, byte)
            heapq.heappush(heap, node)
        return heap

    def merge_nodes(self, heap):
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = HuffmanNode(node1.freq + node2.freq, left=node1, right=node2)
            heapq.heappush(heap, merged)
        return heap

    def make_codes_helper(self, root, current_code):
        if root is None:
            return
        if root.byte is not None:
            self.codes[root.byte] = current_code
            self.reverse_mapping[current_code] = root.byte
            return
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self, heap):
        root = heapq.heappop(heap)
        self.make_codes_helper(root, "")

    def get_encoded_data(self, data_bytes):
        encoded_bits = "".join(self.codes[b] for b in data_bytes)
        extra_padding = 8 - len(encoded_bits) % 8
        encoded_bits += "0" * extra_padding
        padded_info = f"{extra_padding:08b}"
        return padded_info + encoded_bits

    def get_byte_array(self, padded_encoded_bits):
        if len(padded_encoded_bits) % 8 != 0:
            raise ValueError("Encoded bits length is not padded to full bytes.")
        return bytearray(int(padded_encoded_bits[i:i+8], 2) for i in range(0, len(padded_encoded_bits), 8))

    def compress(self, input_path, output_path):
        with open(input_path, 'rb') as file:
            data = file.read()

        freq_dict = self.build_frequency_dict(data)
        heap = self.build_heap(freq_dict)
        heap = self.merge_nodes(heap)
        self.make_codes(list(heap))

        padded_encoded_bits = self.get_encoded_data(data)
        b_arr = self.get_byte_array(padded_encoded_bits)

        with open(output_path, 'wb') as output:
            mapping_size = len(self.codes)
            output.write(mapping_size.to_bytes(2, 'big'))
            for byte, code in self.codes.items():
                output.write(bytes([byte]))
                code_length = len(code)
                output.write(bytes([code_length]))
                output.write(int(code, 2).to_bytes((code_length + 7) // 8, 'big'))
            output.write(b_arr)

        print(f"Compressed file written to {output_path}")

    def remove_padding(self, padded_encoded_bits):
        padded_info = padded_encoded_bits[:8]
        extra_padding = int(padded_info, 2)
        padded_encoded_bits = padded_encoded_bits[8:]
        if extra_padding:
            return padded_encoded_bits[:-extra_padding]
        return padded_encoded_bits

    def decompress(self, input_path, output_path):
        with open(input_path, 'rb') as file:
            bit_data = file.read()

        mapping_size = int.from_bytes(bit_data[:2], 'big')
        idx = 2
        self.codes.clear()
        self.reverse_mapping.clear()
        for _ in range(mapping_size):
            byte = bit_data[idx]
            idx += 1
            code_length = bit_data[idx]
            idx += 1
            num_bytes = (code_length + 7) // 8
            code_bytes = bit_data[idx:idx+num_bytes]
            idx += num_bytes
            code_int = int.from_bytes(code_bytes, 'big')
            code = f"{code_int:0{code_length}b}"
            self.codes[byte] = code
            self.reverse_mapping[code] = byte

        encoded_bytes = bit_data[idx:]
        bit_string = "".join(f"{b:08b}" for b in encoded_bytes)
        actual_bits = self.remove_padding(bit_string)

        current_code = ""
        decoded_bytes = bytearray()
        for bit in actual_bits:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_bytes.append(self.reverse_mapping[current_code])
                current_code = ""

        with open(output_path, 'wb') as output:
            output.write(decoded_bytes)

        print(f"Decompressed file written to {output_path}")

if __name__ == "__main__":
    h = HuffmanCoding()
    import sys
    if len(sys.argv) == 3:
        input_file, output_file = sys.argv[1], sys.argv[2]
        h.compress(input_file, output_file)
    elif len(sys.argv) == 4 and sys.argv[1] == '--decompress':
        h.decompress(sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  Compress: python huffman.py input_file output_file")
        print("  Decompress: python huffman.py --decompress compressed_file output_file")
