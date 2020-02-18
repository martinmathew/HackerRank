import sun.reflect.generics.tree.Tree;

import java.util.*;


class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}


/**
 * 5
 * / \
 * 2   7
 * /
 * 1
 */

class BSTIterator {
    private TreeNode root;
    private Stack<TreeNode> stkNode;

    public BSTIterator(TreeNode root) {
        this.root = root;
        stkNode = new Stack<>();
        init(root);
    }

    private void init(TreeNode node) {
        TreeNode curr = node;
        while (curr != null) {
            stkNode.push(curr);
            curr = curr.left;
        }

    }

    /**
     * @return the next smallest number
     */
    public int next() {
        if (hasNext()) {
            TreeNode node = stkNode.pop();
            init(node.right);
            return node.val;
        }
        return -1;

    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !stkNode.isEmpty();
    }
}


public class Solution {

    boolean solved = false;

    public int romanToInt(String s) {
        Map<Character, Integer> strMap = new HashMap<>();
        strMap.put('I', 1);
        strMap.put('V', 5);
        strMap.put('X', 10);
        strMap.put('L', 50);
        strMap.put('C', 100);
        strMap.put('D', 500);
        strMap.put('M', 1000);

        int res = 0;
        char[] chArr = s.toCharArray();
        for (int i = 0; i < chArr.length; ) {
            if (i < chArr.length - 1 && (strMap.get(chArr[i]) < strMap.get(chArr[i + 1]))) {
                res += (strMap.get(chArr[i + 1]) - strMap.get(chArr[i]));
                i++;
            } else {
                res += (strMap.get(chArr[i]));
            }
            i++;
        }
        return res;
    }


    public String longestCommonPrefix(String[] strs) {
        String str = "";
        if (strs.length == 0) {
            return str;
        }
        str = strs[0];
        for (int i = 1; i < strs.length; ) {
            if (str.length() == 0) {
                return str;
            }
            if (!strs[i].startsWith(str)) {
                str = str.substring(0, str.length() - 1);
            } else {
                i++;
            }
        }
        return str;
    }


    public List<List<Integer>> threeSum(int[] nums) {
        if (nums.length < 2) {
            return new ArrayList<List<Integer>>();
        } else {
            List<List<Integer>> res = new ArrayList<>();
            Arrays.sort(nums);
            for (int i = 0; i < nums.length; i++) {
                if (i == 0 || (i > 0 && nums[i] != nums[i - 1])) {
                    int sum = nums[i] * -1;
                    int low = i + 1;
                    int high = nums.length - 1;
                    while (low < high) {
                        if ((nums[low] + nums[high]) == sum) {
                            List<Integer> li = new ArrayList<>();
                            li.add(nums[i]);
                            li.add(nums[low]);
                            li.add(nums[high]);
                            res.add(li);
                            while (low < high && nums[low] == nums[low + 1]) {
                                low++;
                            }
                            while (low < high && nums[high] == nums[high - 1]) {
                                high--;
                            }
                            low++;
                            high--;
                        } else if (nums[low] + nums[high] < sum) {
                            low++;
                        } else {
                            high--;
                        }
                    }
                }

            }

            return res;
        }

    }

    private List<String> outputStr = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        generateParent(0, 0, n, "");
        return outputStr;
    }


    public void generateParent(int open, int closed, int count, String str) {
        if (str.length() == count * 2) {
            outputStr.add(str);
        } else {
            if (open < count) {
                generateParent(open + 1, closed, count, str + "(");
            }
            if (open > closed) {
                generateParent(open, closed + 1, count, str + ")");
            }

        }

    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) {
            return null;
        }
        if (lists.length == 1) {
            return lists[0];
        } else {
            ListNode[] nums = new ListNode[lists.length];
            int index = 0;
            for (ListNode node : lists) {
                if (node == null) {
                    index++;
                    continue;
                }
                nums[index++] = node;
            }
            ListNode res = null;
            ListNode head = null;
            Arrays.sort(nums, new Comparator<ListNode>() {
                @Override
                public int compare(ListNode o1, ListNode o2) {
                    return o1.val - o2.val;
                }
            });
            while (nums[0] != null) {
                if (res == null) {
                    res = new ListNode(nums[0].val);
                    head = res;
                } else {
                    res.next = new ListNode(nums[0].val);
                    res = res.next;
                }
                int i = 0;
                while (i < lists.length && (nums[i] != null && nums[i++].next == null)) ;
                ListNode nextNode = nums[i - 1].next;
                i = 1;
                while (i < nums.length && (nextNode == null || nextNode.val >= nums[i].val)) {
                    nums[i - 1] = nums[i];
                    i++;
                }
                nums[i - 1] = nextNode;
            }
            return head;
        }


    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    private static ListNode createListNode(int[] arr) {
        ListNode head = null;
        ListNode temp = null;
        for (int num : arr) {
            if (head == null) {
                head = new ListNode(num);
                temp = head;
            } else {
                ListNode nd = new ListNode(num);
                temp.next = nd;
                temp = temp.next;
            }
        }
        return head;
    }

    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> intList = new ArrayList<>();
        if (words.length == 0 || s.length() == 0) {
            return intList;
        }
        int singleWorldLen = words[0].length();
        int arrayWorldLen = singleWorldLen * words.length;
        if (arrayWorldLen > s.length()) {
            return intList;
        }
        Map<String, Integer> strLenMap = new HashMap<>();
        for (String str : words) {
            if (strLenMap.containsKey(str)) {
                int count = strLenMap.get(str);
                strLenMap.put(str, ++count);
            } else {
                strLenMap.put(str, 1);
            }
        }
        for (int i = 0; i < s.length() - arrayWorldLen; i++) {
            int j = i;
            Map<String, Integer> tempMap = new HashMap<>(strLenMap);

            while (j < i + arrayWorldLen) {
                String subStr = s.substring(j, j + singleWorldLen);

                if (tempMap.containsKey(subStr)) {
                    int count = tempMap.get(subStr);
                    if (count == 1) {
                        tempMap.remove(subStr);
                    } else {
                        tempMap.put(subStr, --count);
                    }
                }
                if (tempMap.size() == 0) {
                    break;
                }
                j++;
            }

            if (tempMap.size() == 0) {
                intList.add(i);
            }

        }
        return intList;

    }

    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i + 1] <= nums[i]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    private void swap(char[] nums, int i, int j) {
        char temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    public int longestValidParentheses(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        char[] chArr = s.toCharArray();
        Stack<Character> stack = new Stack<>();
        int maxLeng = 0;
        int len = 0;
        for (int i = 0; i < chArr.length; i++) {
            char ch = chArr[i];
            if (ch == '(') {
                stack.push(ch);
            }
            if (ch == ')') {
                if (stack.empty()) {
                    len = 0;
                    continue;
                }
                stack.pop();
                len += 2;
                maxLeng = Math.max(len, maxLeng);
            }
        }
        return maxLeng;
    }


    public int search(int[] nums, int target) {
        int pivot = findPivot(nums, 0, nums.length - 1);
        if (pivot >= 0 && pivot < nums.length && target == nums[pivot]) {
            return pivot;
        } else if (pivot > 0 && pivot < nums.length && target >= nums[0] && target <= nums[pivot - 1]) {
            return binarySearch(nums, target, 0, pivot - 1);
        } else if (pivot >= 0 && pivot < nums.length - 1 && target >= nums[pivot + 1] && target <= nums[nums.length - 1]) {
            return binarySearch(nums, target, pivot + 1, nums.length - 1);
        }
        return -1;

    }

    private int binarySearch(int[] nums, int target, int start, int end) {
        if (start <= end) {
            int mid = (start + end) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                return binarySearch(nums, target, start, mid - 1);
            } else if (nums[mid] < target) {
                return binarySearch(nums, target, mid + 1, end);
            }
        }
        return -1;
    }


    private int findPivot(int[] nums, int start, int end) {
        if (start <= end) {
            int mid = (start + end) / 2;
            if (mid > 0 && nums[mid - 1] > nums[mid]) {
                return mid;
            } else if (mid < nums.length - 1 && nums[mid] > nums[mid + 1]) {
                return mid + 1;
            } else if (nums[mid] > nums[end]) {
                return findPivot(nums, mid + 1, end);
            } else if (nums[mid] < nums[start]) {
                return findPivot(nums, start, mid - 1);
            }
        }
        return 0;
    }


    public boolean isValidSudoku(char[][] board) {

        Set<Character> rows[] = new Set[9];
        Set<Character> cols[] = new Set[9];
        Set<Character> subBoxes[] = new Set[9];
        for (int i = 0; i < 9; i++) {
            rows[i] = new HashSet<>();
            cols[i] = new HashSet<>();
            subBoxes[i] = new HashSet<>();
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                char ch = board[i][j];
                int subBoxIndex = (i / 3) * 3 + j / 3;
                if (board[i][j] != '.') {
                    if (rows[i].contains(ch) || cols[j].contains(ch) || subBoxes[subBoxIndex].contains(ch)) {
                        return false;
                    }
                    rows[i].add(ch);
                    cols[j].add(ch);

                    subBoxes[subBoxIndex].add(ch);
                }
            }
        }
        return true;
    }

    public void solveSudoku(char[][] board) {
        Set<Character> rows[] = new Set[9];
        Set<Character> cols[] = new Set[9];
        Set<Character> subBoxes[] = new Set[9];
        for (int i = 0; i < 9; i++) {
            rows[i] = new HashSet<>();
            cols[i] = new HashSet<>();
            subBoxes[i] = new HashSet<>();
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                char ch = board[i][j];
                if (board[i][j] != '.') {
                    placeNumber(board, i, j, rows, cols, subBoxes, board[i][j]);
                }
            }
        }
        solveSudokuUtil(board, 0, 0, rows, cols, subBoxes);
    }

    private boolean canPlace(int row, int column, Set<Character> rows[], Set<Character> cols[], Set<Character> boxes[], char number) {
        int boxIndex = (row / 3) * 3 + column / 3;
        return !(rows[row].contains(number) || cols[column].contains(number) || boxes[boxIndex].contains(number));
    }

    public void placeNumber(char[][] board, int row, int col, Set<Character> rows[], Set<Character> cols[], Set<Character> boxes[], char number) {
        board[row][col] = number;
        int boxIndex = (row / 3) * 3 + col / 3;
        rows[row].add(number);
        cols[col].add(number);
        boxes[boxIndex].add(number);
    }

    public void removenumber(char[][] board, int row, int col, Set<Character> rows[], Set<Character> cols[], Set<Character> boxes[], char number) {
        board[row][col] = '.';
        int boxIndex = (row / 3) * 3 + col / 3;
        rows[row].remove(number);
        cols[col].remove(number);
        boxes[boxIndex].remove(number);

    }


    public void placeNextNumber(char[][] board, int row, int col, Set<Character> rows[], Set<Character> cols[], Set<Character> boxes[]) {
        if (row == 8 && col == 8) {
            solved = true;
        } else if (col == 8) {
            solveSudokuUtil(board, row + 1, 0, rows, cols, boxes);
        } else {
            solveSudokuUtil(board, row, col + 1, rows, cols, boxes);
        }
    }


    public void solveSudokuUtil(char[][] board, int row, int col, Set<Character> rows[], Set<Character> cols[], Set<Character> boxes[]) {
        if (board[row][col] == '.') {
            for (char i = 1 + '0'; i < 10 + '0'; i++) {
                if (canPlace(row, col, rows, cols, boxes, i)) {
                    placeNumber(board, row, col, rows, cols, boxes, i);
                    placeNextNumber(board, row, col, rows, cols, boxes);
                    if (!solved) {
                        removenumber(board, row, col, rows, cols, boxes, i);
                    }
                }
            }
        } else {
            placeNextNumber(board, row, col, rows, cols, boxes);
        }
    }


    List<List<Integer>> numList = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        combSumUtil(candidates, target, new ArrayList<Integer>(), 0);
        return numList;
    }


    public void combSumUtil(int[] candidates, int target, List<Integer> sol, int index) {
        if (target == 0) {
            numList.add(new ArrayList<>(sol));
        } else {
            List<Integer> nums = new ArrayList<>();
            for (int i = index; i < candidates.length; i++) {
                if (target >= candidates[i]) {
                    sol.add(new Integer(candidates[i]));
                    combSumUtil(candidates, target - candidates[i], sol, i);
                    sol.remove(new Integer(candidates[i]));
                }
            }
        }
    }


    public void combSumUtil2(int[] candidates, int target, List<Integer> sol, int index) {
        if (target == 0) {
            numList.add(new ArrayList<>(sol));
        } else {
            List<Integer> nums = new ArrayList<>();
            for (int i = index; i < candidates.length; i++) {
                if (i > index && candidates[i] == candidates[i - 1]) {
                    continue;
                }
                if (target >= candidates[i]) {
                    sol.add(new Integer(candidates[i]));
                    combSumUtil2(candidates, target - candidates[i], sol, i + 1);
                    sol.remove(new Integer(candidates[i]));
                }
            }
        }
    }


    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        combSumUtil2(candidates, target, new ArrayList<Integer>(), 0);
        return numList;
    }


    public int firstMissingPositive1(int[] nums) {

        boolean is1Present = false;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                is1Present = true;
            }
            if (nums[i] <= 0 || nums[i] > nums.length) {
                nums[i] = 1;
            }
        }
        if (!is1Present) {
            return 1;
        }
        if (nums.length == 1) {
            return 2;
        }
        for (int i = 0; i < nums.length; i++) {
            if (Math.abs(nums[i]) == nums.length) {
                nums[0] = Math.abs(nums[0]) * -1;
            } else {

                nums[Math.abs(nums[i])] = -1 * Math.abs(nums[Math.abs(nums[i])]);
            }
        }

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > 0) {
                return i;
            }
        }
        if (nums[0] > 0) {
            return nums.length;
        }
        return nums.length + 1;

    }


    public int firstMissingPositive(int[] nums) {
        int n = nums.length;

        // Base case.
        int contains = 0;
        for (int i = 0; i < n; i++)
            if (nums[i] == 1) {
                contains++;
                break;
            }

        if (contains == 0)
            return 1;

        // nums = [1]
        if (n == 1)
            return 2;

        // Replace negative numbers, zeros,
        // and numbers larger than n by 1s.
        // After this convertion nums will contain
        // only positive numbers.
        for (int i = 0; i < n; i++)
            if ((nums[i] <= 0) || (nums[i] > n))
                nums[i] = 1;

        // Use index as a hash key and number sign as a presence detector.
        // For example, if nums[1] is negative that means that number `1`
        // is present in the array.
        // If nums[2] is positive - number 2 is missing.
        for (int i = 0; i < n; i++) {
            int a = Math.abs(nums[i]);
            // If you meet number a in the array - change the sign of a-th element.
            // Be careful with duplicates : do it only once.
            if (a == n)
                nums[0] = -Math.abs(nums[0]);
            else
                nums[a] = -Math.abs(nums[a]);
        }

        // Now the index of the first positive number
        // is equal to first missing positive.
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0)
                return i;
        }

        if (nums[0] > 0)
            return n;

        return n + 1;
    }


    public int trap(int[] height) {

        int index = 0;
        int water = 0;
        int i = 1;
        while (i < height.length) {
            int deduct = 0;
            while (i < height.length && height[i] <= height[index]) {
                deduct += height[i];
                i++;

            }
            if (i < height.length) {
                water += (i - index - 1) * Math.min(height[index], height[i]);
                water -= deduct;
                index = i;
                i++;
            } else if (i == height.length && height[i - 1] >= height[i - 2]) {
                water += (i - index - 1) * Math.min(height[index], height[i - 1]);
                water -= deduct;
            } else {
                water += (i - index - 2) * Math.min(height[index], height[i - 2]);
                water -= deduct - height[i - 1];
            }
        }
        return water;
    }


    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int numZeros = 0;
        List<String> temp = new ArrayList<>();

        for (int i = num1.length() - 1; i >= 0; i--) {


            int div = 0;
            StringBuilder str = new StringBuilder();
            for (int l = 0; l < numZeros; l++) {
                str = str.append("0");
            }
            for (int k = num2.length() - 1; k >= 0; k--) {
                int mul = (num2.charAt(k) - 48) * (num1.charAt(i) - 48) + div;
                div = mul / 10;
                str.insert(0, mul % 10);


            }
            if (div > 0) {
                str.insert(0, div);
            }
            temp.add(str.toString());
            numZeros++;
        }
        int end = 0;
        StringBuilder ans = new StringBuilder();
        int div = 0;
        for (int i = 0; i < temp.get(temp.size() - 1).length(); i++) {
            int sum = 0;
            for (String str : temp) {
                if (str.length() - end - 1 >= 0) {
                    sum += (str.charAt(str.length() - end - 1) - 48);
                }
            }
            int mod = (sum + div) % 10;
            div = (sum + div) / 10;
            ans.insert(0, mod);
            end++;

        }
        if (div > 0) {
            ans.insert(0, div);
        }

        return ans.toString();
    }


    public boolean isMatch(String s, String p) {
        if (s.equals(p) || p.equals("*")) {
            return true;
        }
        if (s.isEmpty() || p.isEmpty()) {
            return false;
        }


        boolean[][] patt_mat = new boolean[s.length() + 1][p.length() + 1];
        patt_mat[0][0] = true;

        for (int pi = 1; pi < p.length() + 1; pi++) {
            if (p.charAt(pi - 1) == '*') {
                int si = 1;
                while (!patt_mat[si - 1][pi - 1] && si < s.length() + 1) {
                    si++;
                }
                patt_mat[si - 1][pi] = patt_mat[si - 1][pi - 1];
                while (si < s.length() + 1) {
                    patt_mat[si][pi] = true;
                    si++;
                }
            } else if (p.charAt(pi - 1) == '?') {
                for (int si = 1; si < s.length() + 1; si++) {
                    patt_mat[si][pi] = patt_mat[si - 1][pi - 1];
                }

            } else {
                for (int si = 1; si < s.length() + 1; si++) {
                    patt_mat[si][pi] = patt_mat[si - 1][pi - 1] && s.charAt(si - 1) == p.charAt(pi - 1);
                }
            }
        }
        return patt_mat[s.length()][p.length()];
    }


    public int jump(int[] nums) {

        if (nums.length <= 1 || nums[0] == 0) {
            return 0;
        }

        int index = 0;
        int count = 0;
        while (index + nums[index] < nums.length - 1) {
            int step = -1;
            int max = -1;
            int next = 0;
            for (int i = 1; i <= nums[index]; i++) {
                int j = index + i;
                step = nums[j];
                if (step + i > max) {
                    max = step + i;
                    next = j;
                }
            }
            index = next;
            count++;
        }
        return count + 1;
    }


    private int jump(int[] nums, int index, int[] arr) {
        if (arr[index] > -1) {
            return arr[index];
        }
        if (index == nums.length - 1) {
            return 0;
        } else if (nums[index] == 0) {
            return 9999999;
        } else {
            int min = Integer.MAX_VALUE;
            for (int i = nums[index]; i > 0; i--) {
                if (index + i < nums.length) {
                    min = Math.min(min, jump(nums, index + i, arr) + 1);
                }
            }
            arr[index] = min;
            return min;
        }
    }

    private void backtrack(int[] nums, int index, List<List<Integer>> out) {
        if (index == nums.length) {
            List<Integer> li = new ArrayList<>();
            for (int num : nums) {
                li.add(num);
            }
            out.add(li);
        } else {
            Set<Integer> used = new HashSet<>();
            for (int i = index; i < nums.length; i++) {
                if (used.add(nums[i])) {
                    swap(nums, i, index);
                    backtrack(nums, index + 1, out);
                    swap(nums, i, index);
                }
            }
        }
    }


    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> out = new ArrayList<>();
        backtrack(nums, 0, out);
        return out;
    }


    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> out = new ArrayList<>();
        backtrack(nums, 0, out);
        return out;
    }


    public void rotate(int[][] matrix) {
        int len = matrix.length - 1;
        for (int i = 0; i <= len / 2; i++) {
            for (int j = i; j < matrix.length - 1 - i; j++) {
                int temp = matrix[j][len - i];
                matrix[j][len - i] = matrix[i][j];
                int temp1 = matrix[len - i][len - j];
                matrix[len - i][len - j] = temp;
                temp = matrix[len - j][i];
                matrix[len - j][i] = temp1;
                matrix[i][j] = temp;
            }
        }

    }


    public List<List<String>> groupAnagrams(String[] strs) {
        int[] prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199};
        Map<Integer, List<String>> res = new HashMap<>();
        for (String str : strs) {
            int hash = getHash(str.toCharArray(), prime);
            if (res.containsKey(hash)) {
                List<String> grp = res.get(hash);
                grp.add(str);
            } else {
                List<String> grp = new ArrayList<>();
                grp.add(str);
                res.put(hash, grp);
            }
        }
        List<List<String>> result = new ArrayList<>();
        for (List col : res.values()) {
            result.add(new ArrayList<>(col));
        }
        return result;
    }

    int getHash(char[] str, int[] prime) {
        int sign = 0;
        for (char ch : str) {
            sign *= prime[ch - 97];
        }
        return sign;
    }

    public double myPow(double x, int n) {
        if (x == 0) {
            return 0;
        }
        if (x == 1) {
            return 1;
        }
        if (n < 0) {
            return myPow(1.0 / x, Math.abs(n));
        }
        if (n == 0) {
            return 1;
        } else if (n % 2 == 0) {
            double val = myPow(x, n / 2);
            return val * val;
        } else {
            return myPow(x, n - 1) * x;
        }

    }


    public boolean canJump(int[] nums, int index) {
        if (nums.length == 1) {
            return true;
        }
        Stack<Integer> stk = new Stack<>();
        stk.push(index);
        while (!stk.isEmpty()) {
            int ind = stk.pop();
            for (int i = 1; i <= nums[ind]; i++) {
                if (ind + i == nums.length - 1) {
                    return true;
                } else if (ind + i > nums.length - 1 || nums[ind + i] == 0) {
                    continue;
                } else {
                    stk.push(ind + i);
                }
            }

        }
        return false;

    }

    public boolean canJump(int[] nums) {
        boolean[] jumps = new boolean[nums.length];
        jumps[nums.length - 1] = true;
        for (int i = nums.length - 2; i >= 0; i--) {
            int farthest = Math.min(i + nums[i], nums.length - 1);
            for (int j = i + 1; j <= farthest; j++) {
                if (jumps[j]) {
                    jumps[i] = true;
                    break;
                }
            }
        }
        return jumps[0];
    }


    class Tuple<T> {
        T start;
        T end;

        Tuple(T start, T end) {
            this.start = start;
            this.end = end;
        }
    }


    public int[][] merge(int[][] intervals) {
        int i = 0;
        int index = 0;
        int[][] res = new int[intervals.length][2];
        while (index < intervals.length) {
            if (i == 0 || res[i - 1][1] < intervals[index][0]) {
                res[i][0] = intervals[index][0];
                res[i][1] = intervals[index][1];
                i++;
            } else {
                res[i - 1][1] = Math.max(res[i - 1][1], intervals[index][1]);
            }

            index++;
        }
        return Arrays.copyOf(res, i);

    }


    public int lengthOfLastWord(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        } else {
            s = s.trim();
            for (int i = s.length() - 1; i >= 0; i--) {
                if (s.charAt(i) == ' ') {
                    return s.length() - 1 - i;
                }
            }
            return s.length();
        }

    }


    public int[][] generateMatrix(int n) {

        int[][] matrix = new int[n][n];
        int r1 = 0, r2 = n - 1;
        int c1 = 0, c2 = n - 1;
        int num = 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; c++) {
                matrix[r1][c] = num++;
            }
            for (int r = r1 + 1; r < r2; r++) {
                matrix[r][c2] = num++;
            }
            if (r1 < r2 && c1 < c2) {
                for (int c = c2; c > c1; c--) {
                    matrix[r2][c] = num++;
                }
                for (int r = r2; r > r1; r--) {
                    matrix[r][c1] = num++;
                }
            }
            r1++;
            r2--;
            c1++;
            c2--;
        }
        return matrix;
    }

    public String getPermutation(int n, int k) {
        int[] factorial = new int[n];
        factorial[0] = 1;
        for (int i = 1; i < n; i++) {
            factorial[i] = factorial[i - 1] * i;
        }
        List<Integer> numList = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            numList.add(i);
        }
        StringBuilder sb = new StringBuilder();
        while (k > 0) {
            int len = numList.size();
            int index = (k - 1) / factorial[len - 1];
            sb.append(numList.get(index));
            numList.remove(index);
            k = k % factorial[len - 1];
        }
        for (int i = numList.size() - 1; i >= 0; i--) {
            sb.append(numList.get(i));
        }
        return sb.toString();
    }

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode(int x) { val = x; }
     * }
     */
    public ListNode rotateRight(ListNode head, int k) {
        int n = 0;
        ListNode temp = head;
        while (temp != null) {
            temp = temp.next;
            n++;
        }
        if (n <= 1 || k == 0 || k % n == 0) {
            return head;
        }

        ListNode kth = findKth(head, k, n);
        ListNode kthhead = kth.next;
        kth.next = null;
        temp = kthhead;
        while (temp != null && temp.next != null) {
            temp = temp.next;
        }
        temp.next = head;
        return kthhead;

    }

    ListNode findKth(ListNode node, int k, int n) {


        k = k % n;
        n -= k;
        ListNode kth = node;
        while (n > 1) {
            kth = kth.next;
            n--;
        }
        return kth;
    }


    public int uniquePaths(int m, int n) {
        int[][] matrix = new int[m][n];
        for (int i = 0; i < m; i++) {
            matrix[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            matrix[0][j] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1];
            }
        }
        return matrix[m - 1][n - 1];
    }


    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int[][] matrix = new int[obstacleGrid.length][obstacleGrid[0].length];
        for (int i = 0; i < obstacleGrid.length; i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            } else {
                matrix[i][0] = 1;
            }
        }
        for (int i = 0; i < obstacleGrid[0].length; i++) {
            if (obstacleGrid[0][i] == 1) {
                break;
            } else {
                matrix[0][i] = 1;
            }
        }

        for (int i = 1; i < obstacleGrid.length; i++) {
            for (int j = 1; j < obstacleGrid[i].length; j++) {
                if (obstacleGrid[i][j] == 1) {
                    continue;
                } else {
                    matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1];
                }
            }
        }

        return matrix[obstacleGrid.length - 1][obstacleGrid[0].length - 1];
    }


    public int minPathSum(int[][] grid) {
        int[][] matrix = new int[grid.length][grid[0].length];
        matrix[0][0] = grid[0][0];
        for (int i = 1; i < matrix.length; i++) {
            matrix[i][0] = matrix[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < matrix[0].length; j++) {
            matrix[0][j] = matrix[0][j - 1] + grid[0][j];
        }

        for (int row = 1; row < grid.length; row++) {
            for (int col = 1; col < grid[row].length; col++) {
                matrix[row][col] = Math.min(matrix[row - 1][col], matrix[row][col - 1]) + grid[row][col];
            }
        }
        return matrix[grid.length - 1][grid[0].length - 1];
    }


    public int[] plusOne(int[] digits) {

        int index = digits.length - 1;
        int c0 = 1;
        while (index >= 0) {
            int sum = (digits[index] + c0);
            digits[index] = sum % 10;
            c0 = sum / 10;
            index--;
        }
        if (c0 == 0) {
            return digits;
        } else {
            int[] newA = new int[digits.length + 1];
            newA[0] = c0;
            for (int i = 1; i < newA.length; i++) {
                newA[i] = digits[i - 1];
            }
            return newA;
        }

    }


    public String addBinary(String a, String b) {

        int f = a.length() - 1;
        int s = b.length() - 1;
        int carry = 0;
        StringBuilder sb = new StringBuilder();
        while (f >= 0 || s >= 0 || carry > 0) {
            int a1 = (f >= 0) ? a.charAt(f--) - '0' : 0;
            int b1 = (s >= 0) ? b.charAt(s--) - '0' : 0;
            sb.append((a1 + b1 + carry) % 2);
            carry = (a1 + b1 + carry) / 2;
        }
        return sb.reverse().toString();

    }


    public List<String> fullJustify(String[] words, int maxWidth) {
        int i = 0, j = 0;
        List<String> ans = new ArrayList<>();
        while (i < words.length) {
            StringBuilder sb = new StringBuilder();
            int len = 0;
            while (i < words.length && len < maxWidth) {
                len += words[i].length() + 1;
                i++;
            }
            len--;
            if (len > maxWidth) {
                len -= words[--i].length() - 1;

                int spacesperword = 0;
                int w = 0;
                if (i != j + 1) {
                    spacesperword = (maxWidth - len) / (i - j - 1);
                    w = (maxWidth - len) % (i - j - 1);
                }
                for (int k = j; k < i; k++) {
                    sb.append(words[k]);
                    if (k == i - 1) {
                        break;
                    }
                    for (int sp = 0; sp <= spacesperword; sp++) {
                        sb.append(" ");

                        if (w > 0) {
                            w--;
                            sb.append(" ");
                        }
                    }
                    sb.append(" ");
                }
                j = i;
            } else {
                for (int k = j; k < i; k++) {
                    sb.append(words[k]).append(" ");
                }
                if (sb.length() > maxWidth) {
                    sb.deleteCharAt(sb.length() - 1);
                }
                j = i;
            }
            while (sb.length() < maxWidth) {
                sb.append(" ");
            }
            ans.add(sb.toString());

        }
        return ans;
    }


    public int mySqrt(int x) {
        if (x == 0) {
            return 0;
        } else if (x <= 3) {
            return 1;
        } else {
            int start = 0;
            int end = x;
            int ans = 0;
            while (start <= end) {
                long mid = (start + end) / 2;
                if (mid * mid == (long) x) {
                    return (int) mid;
                } else if (mid * mid > x) {
                    end = (int) mid - 1;
                } else {
                    start = (int) mid + 1;
                    ans = (int) mid;
                }
            }
            return ans;
        }
    }


    public int climbStairs(int n) {
        int[] stair = new int[n + 1];
        stair[0] = 1;
        stair[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            stair[i] = stair[i - 1] + stair[i - 2];
        }
        return stair[n];
    }


    public String simplifyPath(String path) {
        String[] dir = path.split("/");
        Stack<String> dirStruc = new Stack<>();
        for (String folder : dir) {
            switch (folder) {
                case ".":
                case "":
                    break;
                case "..":
                    if (!dirStruc.isEmpty()) {
                        dirStruc.pop();
                    }
                    break;
                default:
                    dirStruc.push(folder);
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("/");
        int index = 0;
        while (index < dirStruc.size()) {
            sb.append(dirStruc.get(index));
            if (index == dirStruc.size() - 1) {
                break;
            }
            sb.append("/");
            index++;
        }

        return sb.toString();
    }


    public int minDistance(String word1, String word2) {
        int[][] matrix = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i <= word1.length(); i++) {
            matrix[i][0] = i;
        }
        for (int j = 0; j <= word2.length(); j++) {
            matrix[0][j] = j;
        }
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(matrix[i - 1][j - 1], Math.min(matrix[i - 1][j], matrix[i][j - 1])) + 1;
                }
            }
        }
        return matrix[word1.length()][word2.length()];
    }


    public void setZeroes(int[][] matrix) {

        boolean isCol = false;
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][0] == 0) {
                isCol = true;
            }
            for (int j = 1; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[i].length; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (isCol) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
        if (matrix[0][0] == 0) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[0][j] = 0;
            }
        }

    }

    public List<List<Integer>> combine(int n, int k) {
        // init first combination
        LinkedList<Integer> nums = new LinkedList<Integer>();
        for (int i = 1; i < k + 1; ++i)
            nums.add(i);
        nums.add(n + 1);

        List<List<Integer>> output = new ArrayList<List<Integer>>();
        int j = 0;
        while (j < k) {
            // add current combination
            output.add(new LinkedList(nums.subList(0, k)));
            // increase first nums[j] by one
            // if nums[j] + 1 != nums[j + 1]
            j = 0;
            while ((j < k) && (nums.get(j + 1) == nums.get(j) + 1))
                nums.set(j, j++ + 1);
            nums.set(j, nums.get(j) + 1);
        }
        return output;
    }


    int n;
    int k;
    List<List<Integer>> output = new ArrayList<>();

    public List<List<Integer>> subsets(int[] nums) {
        this.n = nums.length;
        for (int i = 0; i < nums.length + 1; i++) {
            this.k = i;
            backtracking(new LinkedList<Integer>(), 1, nums);
        }
        return output;
    }

    public void backtracking(LinkedList<Integer> out, int first, int[] nums) {
        if (out.size() == k) {
            output.add(new ArrayList<>(out));
        } else {
            for (int i = first; i < n + 1; i++) {
                out.add(nums[i - 1]);
                backtracking(out, i + 1, nums);
                out.removeLast();
            }
        }
    }


    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        } else {
            ListNode dummy = new ListNode(0);
            dummy.next = head;
            ListNode prev = dummy;
            ListNode headTemp = head;
            boolean delete = false;
            while (headTemp.next != null) {
                if (headTemp.val == headTemp.next.val) {
                    delete = true;
                } else {
                    if (delete) {
                        prev.next = headTemp;
                    }
                    prev = headTemp;
                    delete = false;
                }
                headTemp = headTemp.next;
            }
            if (delete) {
                prev.next = headTemp;
            }
            return dummy.next;
        }

    }

    public int largestRectangleArea(int[] heights) {

        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        for (int i = 0; i < heights.length; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) {
                maxArea = Math.max(maxArea, heights[stack.pop()] * (i - stack.peek() - 1));
            }
            stack.push(i);
        }
        while (!stack.isEmpty()) {
            maxArea = Math.max(maxArea, heights[stack.pop()] * (heights.length - stack.peek() - 1));
        }
        return maxArea;
    }

    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0) {
            return 0;
        }
        int maxArea = 0;
        int[][] dp = new int[matrix.length][matrix[0].length];
        int[] rows = new int[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = (j == 0) ? 1 : dp[i][j - 1] + 1;
                    int width = dp[i][j];
                    for (int k = i; k >= 0; k--) {
                        width = Math.min(width, dp[k][j]);
                        maxArea = Math.max(maxArea, width * (i - k + 1));

                    }
                }
            }
        }
        return maxArea;
    }

    public ListNode partition(ListNode head, int x) {
        ListNode before_head = new ListNode(-1);
        ListNode before = before_head;
        ListNode after_head = new ListNode(-1);
        ListNode after = after_head;
        while (head != null) {
            if (head.val < x) {
                before.next = head;
                before = before.next;
            } else {
                after.next = head;
                after = after.next;
            }
            head = head.next;
        }
        after.next = null;
        before.next = after_head.next;
        return before_head.next;
    }

    public boolean isScramble(String s1, String s2) {
        if (s1.equals(s2)) {
            return true;
        }
        Map<Character, Integer> mp1 = getMaps(s1);
        Map<Character, Integer> mp2 = getMaps(s2);
        if (mp1.size() != mp2.size()) {
            return false;
        }
        for (Map.Entry<Character, Integer> entry : mp1.entrySet()) {
            if (mp2.getOrDefault(entry.getKey(), 0) != entry.getValue()) {
                return false;
            }
        }
        for (int i = 1; i < s1.length(); i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (isScramble(s1.substring(0, i), s2.substring(s2.length() - i)) && isScramble(s1.substring(i), s2.substring(0, s2.length() - i))) {
                return true;
            }
        }
        return false;
    }


    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int index = m + n - 1;
        int first = m - 1;
        int second = n - 1;
        while (first >= 0 && second >= 0) {
            if (nums1[first] >= nums2[second]) {
                nums1[index] = nums1[first];
                first--;
            } else {
                nums1[index] = nums2[second];
                second--;
            }
            index--;
        }
        if (first < 0 && second >= 0) {
            for (int j = second; j >= 0; j--) {
                nums1[index] = nums2[j];
                index--;
            }
        }

    }


    private Map<Character, Integer> getMaps(String s1) {
        Map<Character, Integer> mp = new HashMap<>();
        for (char ch : s1.toCharArray()) {
            mp.put(ch, mp.getOrDefault(ch, 1) + 1);
        }
        return mp;
    }

    public List<Integer> grayCode(int n) {
        List<Integer> list = new ArrayList<>();
        list.add(0);
        if (n == 0) {
            return list;
        }
        list.add(1);
        int add = 2;
        for (int i = 1; i < n; i++) {
            for (int k = list.size() - 1; k >= 0; k--) {
                list.add(list.get(k) + add);
            }
            add *= 2;
        }

        return list;
    }


    public int numDecodings(String s) {
        if (s.length() == 0) {
            return 1;
        }
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for (int i = 2; i <= s.length(); i++) {
            int first = Integer.valueOf(s.substring(i - 1, i));
            int second = Integer.valueOf(s.substring(i - 2, i));
            if (first >= 1 && first <= 9) {
                dp[i] = dp[i] + dp[i - 1];
            }
            if (second >= 10 && second <= 26) {
                dp[i] = dp[i] + dp[i - 2];
            }
        }
        return dp[s.length()];
    }


    public ListNode reverseBetween(ListNode head, int m, int n) {
        int count = 1;
        ListNode mMinus1thNode = null;
        ListNode mthNode = null;
        ListNode nthNode = null;
        ListNode temp = head;
        ListNode prev = null;
        while (temp != null) {
            if (count == m) {
                mMinus1thNode = prev;
                mthNode = temp;
            }
            if (count == n) {
                nthNode = temp;
                break;
            }
            prev = temp;
            temp = temp.next;
            count++;
        }
        ListNode nPlus1Node = null;
        if (nthNode != null) {
            nPlus1Node = nthNode.next;
            nthNode.next = null;
        }
        Tuple<ListNode> tuple = reverse(mthNode);
        if (mMinus1thNode != null) {
            mMinus1thNode.next = tuple.start;
        } else {
            head = tuple.start;
        }
        if (tuple.end != null) {
            tuple.end.next = nPlus1Node;
        }
        return head;
    }


    private Tuple<ListNode> reverse(ListNode root) {
        if (root != null && root.next == null) {
            return new Tuple<>(root, root);
        } else {
            ListNode prev = null;
            ListNode tail = null;
            while (root != null) {
                if (prev == null) {
                    tail = root;
                }
                ListNode next = root.next;
                root.next = prev;
                prev = root;
                root = next;
            }
            return new Tuple(prev, tail);
        }

    }

    List<String> list = new ArrayList<>();

    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.isEmpty() || s.length() < 4 || s.length() > 12) {
            return new ArrayList<>();
        }
        verifyIPAddress(s, 1, "");
        return list;
    }

    private boolean isValid(String seg) {
        int num = Integer.valueOf(seg);
        return seg.charAt(0) != '0' ? num <= 255 && num >= 1 : seg.length() == 1;
    }


    public void verifyIPAddress(String s, int bite, String sb) {
        if (bite == 4 && !s.isEmpty()) {
            if (isValid(s)) {
                list.add(sb + s);
            }
        } else {
            for (int i = 1; i <= 3; i++) {
                if (s.length() < i) {
                    break;
                }
                String sub = s.substring(0, i);
                if (!isValid(sub)) {
                    continue;
                }
                verifyIPAddress(s.substring(i), bite + 1, sb + sub + ".");
            }
        }
    }


    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> list = new ArrayList<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            TreeNode pop = stack.pop();
            list.add(pop.val);
            curr = pop.right;

        }

        return list;
    }


    public static TreeNode createTree(int[] arr) {
        TreeNode root = null;
        for (int ar : arr) {
            if (root == null) {
                root = new TreeNode(ar);
            } else {
                TreeNode prev = null;
                TreeNode node = root;
                while (node != null) {
                    prev = node;
                    if (node.val >= ar) {
                        node = node.left;
                    } else {
                        node = node.right;
                    }
                }
                TreeNode newNode = new TreeNode(ar);
                if (prev.val >= ar) {
                    prev.left = newNode;
                } else {
                    prev.right = newNode;
                }
            }

        }
        return root;
    }


    public List<TreeNode> generateTrees(int n) {

        return generateTrees(1, n);


    }


    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> list = new ArrayList<>();
        if (start > end) {
            list.add(null);
        } else {
            for (int i = start; i <= end; i++) {

                List<TreeNode> leftTrees = generateTrees(start, i - 1);
                List<TreeNode> rightTrees = generateTrees(i + 1, end);
                for (TreeNode leftRoot : leftTrees) {
                    for (TreeNode rightRoot : rightTrees) {
                        TreeNode root = new TreeNode(i);
                        root.left = leftRoot;
                        root.right = rightRoot;
                        list.add(root);
                    }
                }
            }
        }
        return list;
    }


    public int numTrees(int n) {

        int[] g = new int[n + 1];
        g[0] = 1;
        g[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                g[i] += (g[j - 1] * g[i - j]);
            }
        }
        return g[n];
    }


    public boolean isInterLeaved(char[] s1, int index1, char[] s2, int index2, char[] s3, int index3) {
        if (index1 == s1.length && index2 == s2.length && index3 == s3.length) {
            return true;
        } else {
            boolean isInterleaved = false;
            if (index1 < s1.length && index3 < s3.length && s1[index1] == s3[index3]) {
                isInterleaved = isInterLeaved(s1, index1 + 1, s2, index2, s3, index3 + 1);
            }
            if (!isInterleaved && index3 < s3.length && index2 < s2.length && s2[index2] == s3[index3]) {
                return isInterLeaved(s1, index1, s2, index2 + 1, s3, index3 + 1);
            }
            return isInterleaved;
        }
    }


    Stack<TreeNode> nodes = new Stack<>();
    Stack<TreeNode> lowers = new Stack<>();
    Stack<TreeNode> uppers = new Stack<>();


    private void update(TreeNode node, TreeNode lower, TreeNode upper) {
        if (node == null) {
            return;
        }
        nodes.push(node);
        lowers.push(lower);
        uppers.push(upper);
    }

    public void swap(TreeNode a, TreeNode b) {
        int tmp = a.val;
        a.val = b.val;
        b.val = tmp;
    }

    public void recoverTree(TreeNode root) {
        TreeNode curr = root;
        TreeNode pred = null;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode x = null, y = null;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            if (pred != null && pred.val > curr.val) {
                x = curr;
                if (y == null) {
                    y = pred;
                } else {
                    break;
                }
            }

            pred = curr;
            curr = curr.right;
        }
        swap(x, y);
    }


    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        } else if (p == null ^ q == null) {
            return false;
        } else if (p.val != q.val) {
            return false;
        } else {
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }

    }


    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        Stack<TreeNode> stack = new Stack<>();

        stack.push(root.right);
        stack.push(root.left);
        while (!stack.isEmpty()) {
            TreeNode pop1 = stack.pop();
            TreeNode pop2 = stack.pop();
            if (pop1 == null && pop2 == null) {
                continue;
            }
            if (pop1 == null ^ pop2 == null) {
                return false;
            }
            if (pop1.val != pop2.val) {
                return false;
            }
            stack.push(pop1.left);
            stack.push(pop2.right);
            stack.push(pop1.right);
            stack.push(pop2.left);
        }
        return true;
    }

    public boolean isSymmetric(TreeNode node_1, TreeNode node_2) {
        if (node_1 == null && node_2 == null) {
            return true;
        } else if (node_1 == null ^ node_2 == null) {
            return false;
        } else if (node_1.val != node_2.val) {
            return false;
        } else {
            return isSymmetric(node_1.left, node_2.right) && isSymmetric(node_1.right, node_2.left);
        }

    }


    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {

        List<List<Integer>> res = new ArrayList<>();
        zigzag(root, 0, res);
        return res;
    }

    private void zigzag(TreeNode root, int level, List<List<Integer>> list) {
        if (root == null) {
            return;
        }
        if (level >= list.size()) {
            list.add(new ArrayList<>());
        }
        if (level % 2 == 0) {
            list.get(level).add(root.val);
        } else {
            list.get(level).add(0, root.val);
        }
        zigzag(root.left, level + 1, list);
        zigzag(root.right, level + 1, list);
    }


    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
        }

    }


    private List<Integer> getListValues(List<TreeNode> list) {
        List<Integer> intList = new ArrayList<>();
        for (TreeNode node : list) {
            intList.add(node.val);
        }
        return intList;
    }


    public TreeNode buildTree1(int[] preorder, int[] inorder) {

        if (preorder.length == 0 || inorder.length == 0) {
            return null;
        } else {
            TreeNode root = new TreeNode(preorder[0]);
            int index = findIndexInorder(inorder, preorder[0]);
            int[] leftPreorder = Arrays.copyOfRange(preorder, 1, 1 + index);
            int[] rightPreOrder = Arrays.copyOfRange(preorder, 1 + index, preorder.length);
            int[] leftInorder = Arrays.copyOfRange(inorder, 0, index);
            int[] rightOrder = Arrays.copyOfRange(inorder, index + 1, inorder.length);
            root.left = buildTree1(leftPreorder, leftInorder);
            root.right = buildTree1(rightPreOrder, rightOrder);
            return root;

        }

    }

    int findIndexInorder(int[] inorder, int target) {
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == target) {
                return i;
            }
        }
        return -1;
    }

    private int[] inorder;
    private int[] postorder;
    int postIndex = 0;

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        this.inorder = inorder;
        this.postorder = postorder;
        postIndex = postorder.length - 1;
        return buildTree(0, inorder.length - 1);
    }

    public TreeNode buildTree(int left, int right) {
        if (left > right) {
            return null;
        } else {
            TreeNode root = new TreeNode(postorder[postIndex]);
            int index = findIndexInorder(inorder, postorder[postIndex]);
            postIndex--;
            root.right = buildTree(index + 1, right);
            root.left = buildTree(left, index - 1);
            return root;
        }
    }


    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        LinkedList<List<Integer>> res = new LinkedList<>();
        levelOrderBottom(root, 0, res);
        return res;
    }


    public void levelOrderBottom(TreeNode root, int level, LinkedList<List<Integer>> res) {
        if (root == null) {
            return;
        } else {
            if (level >= res.size()) {
                res.addFirst(new ArrayList<>());
            }
            res.get(res.size() - level - 1).add(root.val);
            levelOrderBottom(root.left, level + 1, res);
            levelOrderBottom(root.right, level + 1, res);
        }
    }


    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start <= end) {
            int mid = (start + end) / 2;
            TreeNode root = new TreeNode(nums[mid]);
            root.left = sortedArrayToBST(nums, start, mid - 1);
            root.right = sortedArrayToBST(nums, mid + 1, end);
            return root;
        } else {
            return null;
        }
    }


    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        } else {
            int htleft = maxDepth(root.left);
            int htRight = maxDepth(root.right);
            if (Math.abs(htleft - htRight) > 1) {
                return false;
            } else {
                return isBalanced(root.left) && isBalanced(root.right);
            }
        }

    }


    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else if (root.left == null && root.right == null) {
            return 1;
        } else {
            int min_length = Integer.MAX_VALUE;
            if (root.left == null) {
                return Math.min(min_length, minDepth(root.right)) + 1;
            } else if (root.right == null) {
                return Math.min(min_length, minDepth(root.left)) + 1;
            } else {
                return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
            }

        }

    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null && sum != 0) {
            return res;
        }
        if (root == null && sum == 0) {
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        LinkedList<Integer> path = new LinkedList<>();
        TreeNode curr = root;
        TreeNode prev = null;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                sum -= curr.val;
                stack.push(curr);
                path.add(curr.val);
                curr = curr.left;
            }
            curr = stack.peek();
            if (curr.right != null && curr.right != prev) {
                curr = curr.right;
                continue;
            }
            if (curr.left == null && curr.right == null && sum == 0) {
                res.add(new ArrayList(path));
            }

            stack.pop();
            path.removeLast();
            prev = curr;
            sum += curr.val;
            curr = null;


        }
        return res;
    }

    public void flatten(TreeNode root) {
        root = flatten1(root);
    }

    public TreeNode flatten1(TreeNode root) {
        if (root == null) {
            return null;
        } else {
            TreeNode rootRight = root.right;
            TreeNode rootLeft = root.left;
            root.right = flatten1(rootLeft);
            TreeNode curr = root;
            while (curr.right != null) {
                curr = curr.right;
            }
            curr.right = flatten1(rootRight);
            root.left = null;
            return root;

        }


    }

    public int findMin(List<Integer> list, int start, int end) {
        int min = Integer.MAX_VALUE;
        int minIndex = -1;
        for (int i = start; i <= end; i++) {
            if (list.get(i) < min) {
                min = list.get(i);
                minIndex = i;
            }
        }
        return minIndex;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int[] dp = new int[triangle.size() + 1];
        for (int i = triangle.size() - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }


    public int maxProfit1(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int price : prices) {
            if (price < minPrice) {
                minPrice = price;
            } else if (maxProfit < price - minPrice) {
                maxProfit = price - minPrice;
            }
        }
        return maxProfit;
    }

    public int maxProfit(int[] prices) {

        int i = 1;
        int buy = -1;
        int sell = -1;
        int[] profits = new int[2];
        while (i < prices.length) {
            while (i < prices.length && prices[i] <= prices[i - 1]) {
                i++;
            }
            buy = i - 1;
            while (i < prices.length && prices[i] >= prices[i - 1]) {
                i++;
            }
            sell = i - 1;
            int profit = (prices[sell] - prices[buy]);
            if (profits[0] <= profit) {
                profits[1] = profits[0];
                profits[0] = profit;
            } else if (profits[1] < profit) {
                profits[1] = profit;
            }

        }
        return profits[0] + profits[1];
    }

    int maxPrice = Integer.MIN_VALUE;

    public int maxPathSum1(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            int leftMax = Math.max(maxPathSum1(root.left), 0);
            int rightMax = Math.max(maxPathSum1(root.right), 0);
            maxPrice = Math.max(maxPrice, leftMax + rightMax + root.val);
            int max = Math.max(leftMax, rightMax);
            return max + root.val;
        }
    }

    public int maxPathSum(TreeNode root) {
        maxPathSum1(root);
        return maxPrice;
    }

    public boolean isPalindrome(String s) {

        int start = 0;

        int end = s.length() - 1;
        s = s.toLowerCase();
        while (start < end) {
            while (start < s.length() && !((s.charAt(start) >= 97 && s.charAt(start) <= 122) || (s.charAt(start) >= 48 && s.charAt(start) <= 57))) {
                start++;
            }
            while (end >= 0 && !((s.charAt(end) >= 97 && s.charAt(end) <= 122) || (s.charAt(end) >= 48 && s.charAt(end) <= 57))) {
                end--;
            }
            if (start < s.length() && end >= 0 && s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;

        }
        return true;
    }


    public int reverse(int x) {
        int mul = 10;
        long revnum = 0;
        while (x != 0) {
            int rem = x % 10;
            x = x / 10;
            revnum = revnum * mul + rem;
            if (revnum < Integer.MIN_VALUE || revnum > Integer.MAX_VALUE) {
                return 0;
            }
        }

        return (int) revnum;


    }

    public boolean isValid1(String s) {

        Stack<Character> stack = new Stack<>();
        for (char ch : s.toCharArray()) {
            if (ch == '(' || ch == '{' || ch == '[') {
                stack.push(ch);
            } else if (ch == ')' || ch == '}' || ch == ']') {
                if (stack.isEmpty()) {
                    return false;
                }
                char pop = stack.pop();
                if (ch == ')' && pop != '(') {
                    return false;
                }
                if (ch == '}' && pop != '{') {
                    return false;
                }
                if (ch == ']' && pop != '[') {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }


    public boolean isPalindrome(int x) {

        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }
        int reversed = 0;
        while (x > reversed) {
            reversed = reversed * 10 + x % 10;
            x = x / 10;
        }
        return reversed == x || x == reversed / 10;

    }


    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {

        ListNode dummy = new ListNode(-1);
        ListNode prev = dummy;
        ListNode temp1 = l1;
        ListNode temp2 = l2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                prev.next = temp1;
                temp1 = temp1.next;
            } else {
                prev.next = temp2;
                temp2 = temp2.next;

            }
            prev = prev.next;
        }
        prev.next = (temp1 == null) ? temp2 : temp1;
        return dummy.next;

    }

    public int removeDuplicates(int[] nums) {
        int ptr = 1;
        int ptr1 = 1;
        while (ptr1 < nums.length) {
            while (ptr1 < nums.length && nums[ptr1] == nums[ptr1 - 1]) {
                ptr1++;
            }
            if (ptr1 < nums.length) {
                nums[ptr] = nums[ptr1];
                ptr++;
                ptr1++;
            }

        }
        return ptr;

    }

    public int strStr(String haystack, String needle) {
        if (needle == null || needle.isEmpty()) {
            return 0;
        }
        int[] pp = preProcess(needle.toCharArray());

        int t = 0;
        int p = 0;

        while (t < haystack.length()) {
            if (haystack.charAt(t) == needle.charAt(p)) {
                p++;
                t++;
                if (p == needle.length()) {
                    return t - p;
                }
            } else {
                if (p != 0) {
                    p = pp[p - 1];
                } else {
                    t++;
                }
            }

        }
        return -1;
    }

    public int[] preProcess(char[] needle) {
        int[] pp = new int[needle.length];
        int i = 0;
        int j = 1;
        while (j < needle.length) {
            if (needle[i] == needle[j]) {
                i++;
                pp[j] = i;
                j++;
            } else {
                if (i != 0) {
                    i = pp[i - 1];
                } else {
                    j++;
                }
            }
        }
        return pp;
    }


    public int majorityElement(int[] nums) {

        int count = 0;
        int candidate = 0;
        for (int num : nums) {
            if (count == 0) {
                candidate = num;
                count++;
            } else if (candidate == num) {
                count++;
            } else if (candidate != num) {
                count--;
            }
        }
        return candidate;
    }


    public int titleToNumber(String s) {
        int mul = 1;
        int num = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            num += (mul * (s.charAt(i) - 'A' + 1));
            mul *= 26;
        }
        return num;
    }


    public int trailingZeroes(int n) {

        int count = 0;
        while (n > 0) {
            n /= 5;
            count += n;
        }
        return count;

    }

    public void reverse(int[] nums, int start, int end) {

        while (start >= 0 && end < nums.length && start < end) {
            swap(nums, start, end);
            start++;
            end--;
        }

    }


    public void rotate(int[] nums, int k) {
        if (nums.length <= 1) {
            return;
        }
        k = k % nums.length;
        reverse(nums, 0, nums.length - k - 1);
        reverse(nums, nums.length - k, nums.length - 1);
        reverse(nums, 0, nums.length - 1);

    }


    public void printBits(int n) {
        int pos = 1;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 32; i++) {
            sb.append((n & (pos << i)) > 0 ? "1" : "0");
        }
        System.out.println(sb.toString());

    }


    public int reverseBits1(int n) {

        int reverse = 0;
        int pos = 31;
        while (pos >= 0 && n > 0) {
            if ((n & 1) == 1) {
                reverse = reverse | 1 << pos;
            }
            pos--;
            n = n >>> 1;
        }
        return reverse;
    }


    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }

        return dp[nums.length - 1];
    }

    public int rob(int[] nums, int start) {
        if (nums.length <= start) {
            return 0;
        }
        if (start == nums.length - 1) {
            return nums[start];
        }
        return Math.max(nums[start] + rob(nums, start + 2), nums[start + 1] + rob(nums, start + 3));
    }


    public boolean isHappy(int n) {

        int happy = 0;
        int num = n;
        Set<Integer> cSet = new HashSet<>();
        while (true) {
            while (num > 0) {
                int temp = num % 10;
                happy = happy + (temp * temp);
                num = num / 10;
            }
            if (happy == 1) {
                return true;
            }
            if (cSet.contains(happy)) {
                return false;
            }
            cSet.add(happy);
            num = happy;
            happy = 0;
        }
    }

    /**
     * egg - add
     * ab - aa
     *
     * @param s
     * @param t
     * @return
     */

    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> chMaps = new HashMap<>();
        if (s.length() != t.length()) {
            return false;
        }
        int is = 0;
        int it = 0;
        while (is < s.length() && it < t.length()) {
            char chs = s.charAt(is);
            char cht = t.charAt(it);
            if ((chMaps.containsKey(chs))) {
                if (chMaps.get(chs) != cht) {
                    return false;
                }

            } else {
                if (!chMaps.containsValue(cht)) {
                    chMaps.put(chs, cht);
                } else {
                    return false;
                }


            }
            is++;
            it++;
        }

        return true;

    }


    public ListNode reverseList(ListNode head) {
        ListNode prevNode = null;
        ListNode temp = head;
        while (temp != null) {
            ListNode next = temp.next;
            temp.next = prevNode;
            prevNode = temp;
            temp = next;
        }
        return prevNode;

    }


    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> maps = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (maps.containsKey(nums[i])) {
                if (k >= (i - maps.get(nums[i]))) {
                    return true;
                } else {
                    maps.put(nums[i], i);
                }
            } else {
                maps.put(nums[i], i);
            }
        }
        return false;
    }


    static class MyStack {

        /**
         * Initialize your data structure here.
         */
        private Queue<Integer> firstQ = null;
        private Queue<Integer> secondQ = null;

        public MyStack() {
            firstQ = new LinkedList<>();
            secondQ = new LinkedList<>();
        }

        /**
         * Push element x onto stack.
         */
        public void push(int x) {
            firstQ.add(x);
        }

        /**
         * Removes the element on top of the stack and returns that element.
         */
        public int pop() {
            Queue<Integer> activeQ = firstQ.isEmpty() ? secondQ : firstQ;
            Queue<Integer> passiveQ = firstQ.isEmpty() ? firstQ : secondQ;
            while (!activeQ.isEmpty()) {
                int val = activeQ.poll();
                if (activeQ.isEmpty()) {
                    return val;
                }
                passiveQ.add(val);
            }

            return -1;
        }

        /**
         * Get the top element.
         */
        public int top() {
            Queue<Integer> activeQ = firstQ.isEmpty() ? secondQ : firstQ;
            Queue<Integer> passiveQ = firstQ.isEmpty() ? firstQ : secondQ;
            while (!activeQ.isEmpty()) {
                int val = activeQ.poll();
                passiveQ.add(val);
                if (activeQ.isEmpty()) {
                    return val;
                }

            }

            return -1;
        }

        /**
         * Returns whether the stack is empty.
         */
        public boolean empty() {
            return firstQ.isEmpty() && secondQ.isEmpty();
        }
    }


    static public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return root;
        } else {
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;
            invertTree(root.left);
            invertTree(root.right);
            return root;
        }
    }

    static public boolean isPowerOfTwo(int n) {
        if (n < 1) {
            return false;
        }

        while (n > 1) {
            if (n % 2 == 1) {
                return false;
            }
            n /= 2;
        }
        return true;
    }


    /**
     * 1 -> 2 -> 3 -> 4 -> 5
     * 1 -> 2 -> 3 -> 4 -> 5 -> 6
     *
     * @param head
     * @return
     */
    static public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        Stack<Integer> stack = new Stack<>();
        ListNode slowPointer = head;
        stack.push(slowPointer.val);
        ListNode fastPointer = head.next;
        boolean isEven = true;
        while (fastPointer != null && fastPointer.next != null) {
            fastPointer = fastPointer.next;
            if (fastPointer.next == null) {
                isEven = false;
                break;
            }
            fastPointer = fastPointer.next;
            slowPointer = slowPointer.next;
            stack.push(slowPointer.val);
        }

        if (isEven) {
            slowPointer = slowPointer.next;

        } else {
            slowPointer = slowPointer.next.next;
        }
        while (slowPointer != null & !stack.isEmpty()) {
            if (!stack.pop().equals(slowPointer.val)) {
                return false;
            }
            slowPointer = slowPointer.next;
        }
        return true;


    }


    static boolean isDescendant(TreeNode node, TreeNode child) {
        if (node == null) {
            return false;
        }
        if (node.val == child.val) {
            return true;
        } else {
            return isDescendant(node.left, child) || isDescendant(node.right, child);
        }
    }

    static public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        } else {
            if (root.val >= p.val && root.val <= q.val) {
                return root;
            } else if (root.val >= p.val && root.val >= q.val) {
                return lowestCommonAncestor1(root.left, p, q);
            } else if (root.val <= p.val && root.val <= q.val) {
                return lowestCommonAncestor1(root.right, p, q);
            }
            return null;
        }

    }

    static public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

        if (p.val <= q.val) {
            return lowestCommonAncestor1(root, p, q);
        } else {
            return lowestCommonAncestor1(root, q, p);
        }
    }

    static int[] arr = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199};

    static private int genhash(char[] charr) {
        int hash = 1;
        for (char a : charr) {
            hash *= arr[a - 'a'];
        }
        return hash;
    }

    static public boolean isAnagram(String s, String t) {
        int firstHash = genhash(s.toCharArray());
        int secHash = genhash(t.toCharArray());
        return firstHash == secHash;

    }


    /**
     * 69  6    69
     * 9
     *
     * @param num
     * @return
     */
    public static boolean isStrobogrammatic(String num) {
        Map<Character, Character> strMap = new HashMap<>();
        strMap.put('6', '9');
        strMap.put('9', '6');
        strMap.put('8', '8');
        strMap.put('1', '1');
        strMap.put('0', '0');

        int start = 0;
        int end = num.length() - 1;
        while (start <= end) {
            if (strMap.containsKey(num.charAt(start))) {
                if (strMap.get(num.charAt(start)) != num.charAt(end)) {
                    return false;
                }
                start++;
                end--;
            } else {
                return false;
            }
        }


        return true;
    }


    public boolean canPermutePalindrome(String s) {
        if (s.length() <= 1) {
            return true;
        } else {
            Map<Character, Integer> perMaps = new HashMap<>();
            for (char ch : s.toCharArray()) {
                perMaps.put(ch, perMaps.getOrDefault(ch, 0) + 1);
            }

            Set<Map.Entry<Character, Integer>> mapSet = perMaps.entrySet();
            int oddCount = 0;
            for (Map.Entry<Character, Integer> entry : mapSet) {
                if (entry.getValue() % 2 == 1) {
                    oddCount++;
                }
                if (oddCount > 1) {
                    return false;
                }
            }
        }
        return true;
    }

    static public int missingNumber(int[] nums) {
        int gauss = nums.length * (nums.length - 1) / 2;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        return gauss - sum;
    }

    /**
     * Because we know that nums contains nn numbers and that it is missing exactly one number on the range [0..n-1][0..n−1], we know that nn definitely replaces the missing number in nums.
     * Therefore, if we initialize an integer to nn and XOR it with every index and value, we will be left with the missing number.
     * Consider the following example (the values have been sorted for intuitive convenience, but need not be):
     * <p>
     * Index	0	1	2	3
     * Value	0	1	3	4
     * \begin{aligned} missing &= 4 \wedge (0 \wedge 0) \wedge (1 \wedge 1) \wedge (2 \wedge 3) \wedge (3 \wedge 4) \\ &= (4 \wedge 4) \wedge (0 \wedge 0) \wedge (1 \wedge 1) \wedge (3 \wedge 3) \wedge 2 \\ &= 0 \wedge 0 \wedge 0 \wedge 0 \wedge 2 \\ &= 2 \end{aligned}
     * missing
     * ​
     * <p>
     * =4∧(0∧0)∧(1∧1)∧(2∧3)∧(3∧4)
     * =(4∧4)∧(0∧0)∧(1∧1)∧(3∧3)∧2
     * =0∧0∧0∧0∧2
     * =2
     * ​
     * 2.71
     * <p>
     * 4
     * / \
     * 2   5
     * / \
     * 1   3
     */


    static public int closestValue(TreeNode root, double target) {

        double diff = Double.MAX_VALUE;
        int closest = 0;
        while (root != null) {
            if (diff > Math.abs(((double) root.val - target))) {
                closest = root.val;
                diff = Math.abs(((double) root.val - target));
            }
            root = (double) root.val >= target ? root.left : root.right;

        }
        return closest;
    }


    public List<List<String>> partition(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(i) == s.charAt(j) && ((j - i) <= 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                }
            }
        }
        generateString(s, dp, 0, new ArrayList<>());
        return results;
    }


    List<List<String>> results = new ArrayList<>();

    private void generateString(String s, boolean[][] dp, int i, List<String> res) {
        if (i == s.length()) {
            results.add(new ArrayList<>(res));
        } else {
            for (int p = i; p < s.length(); p++) {
                if (dp[i][p]) {
                    res.add(s.substring(i, p + 1));
                    generateString(s, dp, p + 1, res);
                    res.remove(res.size() - 1);
                }
            }
        }
    }

    private boolean isRepeating(char[] ch) {
        if (ch.length == 1) {
            return false;
        }
        for (int i = 1; i < ch.length; i++) {
            if (ch[0] != ch[i]) {
                return false;
            }
        }
        return true;
    }

    public String fractionToDecimal(int numerator, int denominator) {
        long numer = (long) numerator;
        long den = (long) denominator;

        StringBuilder sb = new StringBuilder();
        if (numer == 0) {
            return "0";
        }
        if (numer < 0 ^ den < 0) {
            sb.append("-");
        }
        numer = Math.abs(numer);
        den = Math.abs(den);
        sb.append(Long.toString(numer / den));
        if (numer % den == 0) {
            return sb.toString();
        }
        sb.append(".");
        numer = Math.abs(numer);
        den = Math.abs(den);
        long rem = numer % den;
        Map<Long, Integer> pos = new HashMap<>();
        while (rem != 0) {
            if (pos.containsKey(rem)) {
                sb.insert(pos.get(rem), "(");
                sb.append(")");
                break;
            }
            pos.put(rem, sb.length());
            rem *= 10;
            long res = rem / den;
            sb.append(res);
            rem = rem % den;
        }
        return sb.toString();
    }

    private class IntegerComparator implements Comparator<IntDigitPair> {


        @Override
        public int compare(IntDigitPair o1, IntDigitPair o2) {
            List<Integer> dig1 = o1.digitList;
            List<Integer> dig2 = o2.digitList;
            int i = 0;
            while (i < dig1.size() && i < dig2.size()) {
                if (dig1.get(i) != dig2.get(i)) {
                    return dig2.get(i) - dig1.get(i);
                }
                i++;
            }
            if (i >= dig1.size() && i >= dig2.size()) {
                return 0;
            } else if (i >= dig2.size() && (i > 0 && dig1.get(i) > dig2.get(i - 1))) {
                return -1;
            } else if (i >= dig1.size() && (i > 0 && dig2.get(i) > dig1.get(i - 1))) {
                return 1;
            } else if (i >= dig2.size()) {
                return 1;
            } else {
                return -1;
            }
        }
    }


    List<Integer> getDigits(int num) {
        List<Integer> digList = new ArrayList<>();
        while (num > 0) {
            int rem = num % 10;
            num = num / 10;
            digList.add(0, rem);
        }
        return digList;
    }


    static class IntDigitPair {
        int num;
        private List<Integer> digitList;

        IntDigitPair(int num, List<Integer> digitList) {
            this.num = num;
            this.digitList = digitList;
        }
    }

    /**
     * [10,2] ->
     *
     * @param nums
     * @return
     */
    public String largestNumber(int[] nums) {
        IntDigitPair[] pairs = new IntDigitPair[nums.length];
        for (int i = 0; i < nums.length; i++) {
            List<Integer> digits = getDigits(nums[i]);
            pairs[i] = new IntDigitPair(nums[i], digits);
        }
        Arrays.sort(pairs, new IntegerComparator());
        StringBuilder sb = new StringBuilder();
        for (IntDigitPair pair : pairs) {
            sb.append(pair.num);
        }
        return sb.toString();
    }


    public void reverseChars(char[] ch, int start, int end) {

        while (start < end) {
            char temp = ch[start];
            ch[start] = ch[end];
            ch[end] = temp;
            start++;
            end--;
        }

    }


    public void reverseWords(char[] s) {

        int start = 0;
        int end = 0;
        while (start < s.length) {

            if (s[start] == ' ') {
                start++;
            } else {
                end = start + 1;
                while (end < s.length && s[end] != ' ') {
                    end++;
                }
                reverseChars(s, start, end - 1);
                start = end;


            }


        }
        reverseChars(s, 0, s.length - 1);

    }


    public List<String> findRepeatedDnaSequences(String s) {

        Set<String> seen = new HashSet<>();
        Set<String> repeated = new HashSet<>();
        int start = 0;
        while (start + 10 <= s.length()) {
            String subStr = s.substring(start, start + 10);
            if (!seen.add(subStr)) {
                repeated.add(subStr);
            }
            start++;
        }
        return new ArrayList<>(repeated);
    }


    private Map<Character, Integer> getCountMaps(String magazine) {
        Map<Character, Integer> countMap = new HashMap<>();
        for (char ch : magazine.toCharArray()) {
            countMap.put(ch, countMap.getOrDefault(ch, 0) + 1);
        }
        return countMap;
    }

    public boolean canConstruct(String ransomNote, String magazine) {
        int[] ascii = new int[128];
        for (char ch : magazine.toCharArray()) {
            ascii[ch]++;
        }
        for (char ch : ransomNote.toCharArray()) {
            if (--ascii[ch] < 0) {
                return false;
            }
        }
        return true;
    }


    public int firstUniqChar(String s) {
        int[] ascii = new int[128];
        for (char ch : s.toCharArray()) {
            ascii[ch]++;
        }
        for (int i = 0; i < s.length(); i++) {
            if (ascii[s.charAt(i)] == 1) {
                return i;
            }
        }
        return -1;
    }

    public char findTheDifference(String s, String t) {
        int[] ascii = new int[128];
        for (char ch : s.toCharArray()) {
            ascii[ch]++;
        }
        for (int i = 0; i < t.length(); i++) {
            if (--ascii[t.charAt(i)] < 0) {
                return t.charAt(i);
            }
        }
        return '0';

    }

    /**
     * abc
     * ahcgdb
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isSubsequence(String s, String t) {
        Map<Character, List<Integer>> maps = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            if (!maps.containsKey(t.charAt(i))) {
                maps.put(t.charAt(i), new ArrayList<>());
            }
            maps.get(t.charAt(i)).add(i);
        }

        int st = 0;
        for (char ch : s.toCharArray()) {
            if (maps.containsKey(ch)) {
                int index = Collections.binarySearch(maps.get(ch), st);
                if (index == -1) {
                    return false;
                }
                st = index + 1;

            }
        }
        return true;

    }

    public void solve(char[][] board) {
        if (board.length < 2 || board[0].length < 2) {
            return;
        }
        for (int i = 0; i < board.length; i++) {
            if (board[i][0] == 'O') {
                dfs(board, i, 0);
            }
            if (board[i][board[i].length - 1] == 'O') {
                dfs(board, i, board[i].length - 1);
            }
        }
        for (int i = 0; i < board[0].length; i++) {
            if (board[0][i] == 'O') {
                dfs(board, 0, i);
            }
            if (board[board.length - 1][i] == 'O') {
                dfs(board, board.length - 1, i);
            }
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
                if (board[i][j] == '*') {
                    board[i][j] = 'O';
                }
            }
        }

    }


    public void dfs(char[][] board, int row, int col) {
        if (row < 0 || row >= board.length || col < 0 || col >= board[row].length) {
            return;
        } else {
            board[row][col] = '*';
            if (row > 0 && board[row - 1][col] == 'O') {
                dfs(board, row - 1, col);
            }
            if (row < board.length - 2 && board[row + 1][col] == 'O') {
                dfs(board, row + 1, col);
            }
            if (col > 0 && board[row][col - 1] == 'O') {
                dfs(board, row, col - 1);
            }
            if (col < board[row].length - 2 && board[row][col + 1] == 'O') {
                dfs(board, row, col + 1);
            }
        }
    }


    private boolean isPrime(int n) {
        if (n <= 1) {
            return false;
        }
        if (n <= 3) {
            return true;
        }
        if (n % 2 == 0 || n % 3 == 0) {
            return false;
        }

        for (int i = 5; i * i < n; i = i + 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }
        return true;
    }


    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (!notPrime[i]) {
                count++;
                for (int j = 2; i * j < n; j++) {
                    notPrime[i * j] = true;
                }
            }
        }
        return count;
    }


    Stack<Integer> recurse = new Stack<>();
    List<Integer> courses = new ArrayList<>();

    private boolean dfs(Integer course, Map<Integer, Set<Integer>> adjacencyList, boolean[] visited) {
        recurse.add(course);
        visited[course] = true;

        for (int dep : adjacencyList.getOrDefault(course, new HashSet<>())) {
            if (recurse.contains(dep)) {
                return false;
            } else if (visited[dep]) {
                continue;
            } else {
                if (!dfs(dep, adjacencyList, visited)) {
                    return false;
                }
            }
        }

        recurse.remove(course);
        courses.add(course);
        return true;
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (prerequisites.length == 0) {
            int[] cours = new int[numCourses];
            for (int i = 0; i < numCourses; i++) {
                cours[i] = i;
            }
            return cours;
        }
        Map<Integer, Set<Integer>> adjacencyList = new HashMap<>();

        for (int[] preq : prerequisites) {
            if (!adjacencyList.containsKey(preq[0])) {
                adjacencyList.put(preq[0], new HashSet<>());
            }
            adjacencyList.get(preq[0]).add(preq[1]);
        }
        boolean[] visited = new boolean[numCourses];
        for (int i = 0; i < numCourses; i++) {
            if (!visited[i] && !dfs(i, adjacencyList, visited)) {
                return new int[0];
            }

        }
        int[] courseList = new int[courses.size()];
        for (int i = 0; i < courses.size(); i++) {
            courseList[i] = courses.get(i);
        }
        return courseList;
    }


    private boolean search(char[][] board, int row, int col, String word, int index) {
        if (index >= word.length()) {
            return true;
        } else if (board[row][col] != word.charAt(index)) {
            return false;
        } else {
            board[row][col] = '#';
            if ((row < board.length - 1 && search(board, row + 1, col, word, index + 1))
                    || (col < board[row].length - 1 && search(board, row, col + 1, word, index + 1)) ||
                    (row > 0 && search(board, row - 1, col, word, index + 1))
                    || (col > 0 && search(board, row, col - 1, word, index + 1))) {
                board[row][col] = word.charAt(index);

                return true;
            } else {
                board[row][col] = word.charAt(index);

                return false;
            }
        }
    }

    public List<String> findWords(char[][] board, String[] words) {

        TrieNode root = buildTrie(words);
        List<String> list = new ArrayList<>();

        searchOneWord(board, root, list);
        return list;
    }

    private void searchOneWord(char[][] board, TrieNode root, List<String> res) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                searchWord(board, i, j, root, res);


            }
        }

    }

    private void searchWord(char[][] board, int row, int col, TrieNode root, List<String> res) {
        char c = board[row][col];
        if (c == '#') {
            return;
        }
        TrieNode next = root.next[c - 'a'];
        if (next == null) {
            return;
        } else if (next.word != null) {
            res.add(next.word);
            next.word = null;
        }


        board[row][col] = '#';
        if (row > 0) {
            searchWord(board, row - 1, col, next, res);
        }
        if (col > 0) {
            searchWord(board, row, col - 1, next, res);
        }
        if (row < board.length - 1) {
            searchWord(board, row + 1, col, next, res);
        }
        if (col < board[row].length - 1) {
            searchWord(board, row, col + 1, next, res);
        }
        board[row][col] = c;


    }

    private TrieNode buildTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode p = root;
            for (int i = 0; i < word.length(); i++) {
                TrieNode next = p.next[word.charAt(i) - 'a'];
                if (next == null) {
                    p.next[word.charAt(i) - 'a'] = new TrieNode();
                    next = p.next[word.charAt(i) - 'a'];
                }
                p = next;
            }
            p.word = word;
        }
        return root;
    }


    class TrieNode {
        TrieNode[] next = new TrieNode[26];
        String word;
    }


    public int findKthLargest(int[] nums, int k) {
        int index = -1;
        int start = 0;
        int end = nums.length - 1;
        k = nums.length - k;
        if (k < 0) {
            return -1;
        }
        while (start < end) {
            index = partition(nums, start, end);

            if (index == k) {
                return nums[k];
            }
            if (index > k) {

                end = index - 1;

            } else if (index < k) {
                start = index + 1;

            }
        }

        return nums[k];
    }


    public int partition(int[] nums, int start, int end) {

        int pivot = start + (new Random().nextInt(end - start));
        int pivotNum = nums[pivot];
        swap(nums, pivot, end);
        int swapIndex = start;
        for (int i = start; i <= end - 1; i++) {
            if (nums[i] < pivotNum) {
                swap(nums, swapIndex, i);
                swapIndex++;
            }
        }
        swap(nums, swapIndex, end);
        return swapIndex;
    }


    int minLen = Integer.MAX_VALUE;

    public void numSquaresUtil(int n, List<Integer> res) {
        if (n == 0) {
            if (minLen > res.size()) {
                minLen = res.size();
            }
        } else {

            for (int i = 1; i * i <= n; i++) {
                if (i * i <= n) {
                    res.add(i * i);
                    numSquaresUtil(n - (i * i), res);
                    res.remove(new Integer(i * i));
                }
            }

        }

    }

    public int numSquares(int n) {
        numSquaresUtil(n, new ArrayList<>());
        return minLen;
    }


    public int findDuplicate(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                return nums[i];
            }
        }
        return -1;
    }

    /**
     * [
     * [0,1,0],
     * [0,0,1],
     * [1,1,1],
     * [0,0,0]
     * ]
     *
     * @param board
     */


    private boolean isAlive(int[][] board, int row, int col) {
        return row >= 0 && col >= 0 && row < board.length && col < board[row].length && board[row][col] == 1;
    }


    public int minCostClimbingStairs(int[] cost) {

        if (cost.length < 2) {
            return 0;
        }
        int[] minCosts = new int[cost.length + 1];

        for (int i = 2; i <= cost.length; i++) {
            minCosts[i] = Math.min(minCosts[i - 1] + cost[i - 1], minCosts[i - 2] + cost[i - 2]);
        }
        return minCosts[cost.length];
    }


    public int rob1(int[] nums) {
        if (nums.length == 0) {
            return 0;
        } else if (nums.length == 1) {
            return nums[0];
        }

        int[] moneyRobbed = new int[nums.length + 1];
        moneyRobbed[1] = nums[0];
        moneyRobbed[2] = Math.max(nums[0], nums[1]);
        for (int i = 3; i < moneyRobbed.length; i++) {
            moneyRobbed[i] = Math.max(moneyRobbed[i - 1], moneyRobbed[i - 2] + nums[i - 1]);
        }
        return moneyRobbed[nums.length];
    }

    public int numWays(int n, int k) {
        if (n == 0 || k == 0) {
            return 0;
        } else if (n == 1) {
            return k;
        } else {
            int count = 0;
            for (int j = 2; j <= n; j++) {
                count = count + k * (k - 1);
            }
            return count;
        }
    }

    /**
     * 5   3   4   5
     * <p>
     * 5
     * <p>
     * 3
     * <p>
     * 4
     * <p>
     * 5
     *
     * @param piles
     * @return
     */


    public boolean stoneGame(int[] piles) {
        int[][] dp = new int[piles.length + 2][piles.length + 2];

        for (int size = 1; size <= piles.length; size++) {
            for (int i = 0; i + size <= piles.length; i++) {
                int j = i + size - 1;
                boolean isAlexTurn = (i + j + piles.length) % 2 == 1;
                if (isAlexTurn) {
                    dp[i + 1][j + 1] = Math.max(piles[i] + dp[i + 2][j + 1], piles[j] + dp[i + 1][j]);
                } else {
                    dp[i + 1][j + 1] = Math.min(-piles[i] + dp[i + 2][j + 1], -piles[j] + dp[i + 1][j]);
                }
            }
        }
        return dp[1][piles.length] > 0;
    }


    public int mctFromLeafValues(int[] arr) {
        return 0;
    }


    private int minLeaves(int[] arr, int start, int maxLeft, int maxRight) {
        if (arr.length - (start + 1) < 2) {
            return 0;
        } else if (arr.length - (start + 1) == 2) {
            return arr[start] * arr[start + 1];
        } else {
            return Math.min(arr[start], minLeaves(arr, start + 1, maxLeft, maxRight));
        }
    }

    //2,2,3,1
    public int thirdMax(int[] nums) {
        if (nums != null && nums.length == 0) {
            return 0;
        }
        long firstMax = Long.MIN_VALUE;
        long secMax = Long.MIN_VALUE;
        long thirdMax = Long.MIN_VALUE;
        for (int num : nums) {
            if (num > firstMax) {
                thirdMax = secMax;
                secMax = firstMax;
                firstMax = num;
            } else if (num > secMax && num != firstMax) {
                thirdMax = secMax;
                secMax = num;
            } else if (num > thirdMax && num != firstMax && num != secMax) {
                thirdMax = num;
            }
        }
        return thirdMax == Long.MIN_VALUE ? (int) firstMax : (int) thirdMax;
    }

    public boolean validWordSquare(List<String> words) {

        for (int col = 0; col < words.size(); col++) {
            StringBuilder sb = new StringBuilder();
            for (int row = 0; row < words.size(); row++) {
                if (words.get(row).length() > col) {
                    sb.append(words.get(row).charAt(col));
                }
            }
            int index = words.indexOf(sb.toString());
            if (index == -1 || index != col) {
                return false;
            }
        }
        return true;
    }

    class Node {
        public int val;
        public List<Node> children;

        public Node() {
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
    }

    ;

    public List<List<Integer>> levelOrder(Node root) {
        levelOrder(root, 1);
        return res;
    }

    List<List<Integer>> res = new ArrayList<>();

    public void levelOrder(Node root, int level) {
        if (root == null) {
            return;
        } else {
            if (res.size() < level) {
                res.add(new ArrayList<>());
            }
            res.get(level - 1).add(root.val);
            for (Node child : root.children) {
                levelOrder(child, level + 1);
            }

        }

    }

    public int countSegments(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.length() == 0) {
            return 0;
        }

        String segs[] = s.split(" ");
        return segs.length - 1;

    }

    /**
     * [2, 2, 3, 2]
     * 10 & 11
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int num : nums) {
            int count = counts.getOrDefault(num, 0);
            counts.put(num, ++count);
        }

        for (int num : counts.keySet()) {
            if (counts.get(num) == 1) {
                return num;
            }
        }
        return -1;
    }

    /**
     * @return
     */
    public int searchInsert(int[] nums, int target) {

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) {
                return i;
            } else if (nums[i] > target) {
                return i - 1;
            }
        }
        return nums.length;
    }

    /**
     * beginWord = "hit",
     * endWord = "cog",
     * wordList = ["hot","dot","dog","lot","log","cog"]
     *
     * @return
     */

    private int findDistance(char[] word_1, char[] word_2) {
        int distance = 0;
        for (int i = 0; i < word_1.length; i++) {
            if (word_1[i] != word_2[i]) {
                distance++;
            }
        }
        return distance;
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) {
            return 0;
        }
        boolean[] ignore = new boolean[wordList.size()];
        return minLadderWord(beginWord, endWord, wordList, ignore);
    }


    private int minLadderWord(String beginWord, String endWord, List<String> wordList, boolean[] ignore) {
        if (beginWord.equals(endWord)) {
            return 1;
        } else {
            int minLength = 0;
            for (int i = 0; i < wordList.size(); i++) {
                if (!ignore[i]) {
                    int startDistance = findDistance(beginWord.toCharArray(), wordList.get(i).toCharArray());
                    int endDistance = findDistance(wordList.get(i).toCharArray(), endWord.toCharArray());
                    int totalDistance = findDistance(beginWord.toCharArray(), endWord.toCharArray());
                    if (startDistance == 1 && endDistance <= totalDistance) {
                        ignore[i] = true;
                        minLength = Math.min(minLadderWord(wordList.get(i), endWord, wordList, ignore), minLength) + 1;
                        ignore[i] = false;
                    }
                }
            }
            return minLength;
        }
    }

    /**
     * 5 -> 6
     * 11
     * /      \
     * 10      7
     * \     /
     * 9->8
     */

    public ListNode detectCycle(ListNode head) {

        if (head == null) {
            return null;
        } else {
            ListNode slow = head;
            ListNode fast = head.next;
            while (slow != null && fast != null && slow != fast) {
                slow = slow.next;
                if (fast.next != null) {
                    fast = fast.next.next;
                }
            }

            if (slow == null || fast == null) {
                return null;
            }

            ListNode start = head;
            slow = slow.next;
            if (slow == null) {
                return null;
            }
            while (slow != start) {
                slow = slow.next;
                start = start.next;
            }
            return start;

        }

    }


    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        } else {
            ListNode slow = head;
            ListNode fast = head.next;
            while (slow != null && fast != null && slow != fast) {
                slow = slow.next;
                fast = fast.next;
                if (fast != null) {
                    fast = fast.next;
                }
            }
            return slow != null && fast != null;


        }
    }

    public String reverseWords(String s) {
        String[] splits = s.split(" ");
        StringBuffer sb = new StringBuffer();
        for (int i = splits.length - 1; i > 0; i--) {
            if (splits[i].equals("")) {
                continue;
            }
            sb.append(splits[i]).append(" ");
        }
        if (splits.length > 0 && !splits[0].equals("")) {
            sb.append(splits[0]);
        }
        return sb.toString().trim();
    }


    /**
     * Input: [1,2,3,4,5]
     * <p>
     * 10
     * / \
     * 5   15
     * / \
     * 2   7
     * <p>
     * <p>
     * <p>
     * <p>
     * <p>
     * <p>
     * <p>
     * [2,7,5,10,15]
     * <p>
     * 2
     * / \
     * 7   5
     * / \
     * 10  15
     *
     * @param
     * @return
     */

    private TreeNode createUpSideDownBinaryTree(List<TreeNode> res) {

        Queue<TreeNode> q = new LinkedList<>();
        TreeNode root = res.get(0);
        q.add(root);
        int i = 1;
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (i >= res.size()) {
                return root;
            }
            if (i < res.size()) {
                node.right = res.get(i);
                q.add(node.right);
                i++;
            }
            if (i < res.size()) {
                node.left = res.get(i);
                q.add(node.left);
                i++;
            }
        }
        return root;
    }

    static private List<TreeNode> getListUpsideDown(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        } else {
            List<TreeNode> left = getListUpsideDown(root.left);
            List<TreeNode> right = getListUpsideDown(root.right);

            List<TreeNode> res = new ArrayList<>();
            res.addAll(left);
            res.addAll(right);
            res.add(root);

            return res;
        }
    }

    public TreeNode upsideDownBinaryTree(TreeNode root) {
        List<TreeNode> resList = getListUpsideDown(root);
        return createUpSideDownBinaryTree(resList);
    }

    /**
     * Input: nums = [1,2,1,3,5,6,4]
     * Output: 1 or 5
     * Explanation: Your function can return either index number 1 where the peak element is 2,
     * or index number 5 where the peak element is 6.
     *
     * @param nums
     * @return
     */
    public int findPeakElement(int[] nums) {
        long[] newArr = new long[nums.length + 2];
        for (int i = 0; i < nums.length; i++) {
            newArr[i + 1] = nums[i];
        }
//        System.arraycopy(nums,0,newArr,1,nums.length);
        newArr[0] = Long.MIN_VALUE;
        newArr[newArr.length - 1] = Long.MIN_VALUE;
        return findPeakElement(newArr, 0, newArr.length - 1) - 1;
    }


    private int findPeakElement(long[] nums, int start, int end) {
        if (start < end) {
            int mid = (start + end) / 2;
            if (mid > 0 && mid < nums.length - 1 && nums[mid - 1] < nums[mid] && nums[mid + 1] < nums[mid]) {
                return mid;
            }
            if (mid > 0 && nums[mid - 1] < nums[mid]) {
                return findPeakElement(nums, mid, end);
            } else if (mid < nums.length && nums[mid] > nums[mid + 1]) {
                return findPeakElement(nums, start, mid);
            } else {
                int res = findPeakElement(nums, start, mid);
                if (res == -1) {
                    res = findPeakElement(nums, mid, end);
                }
                return res;
            }
        }
        return -1;
    }


    static class TwoSum {
        private List<Integer> store;
        private Map<Integer, Integer> numMap;

        /**
         * Initialize your data structure here.
         */
        public TwoSum() {
            store = new ArrayList<>();
            numMap = new HashMap<>();
        }

        /**
         * Add the number to an internal data structure..
         */
        public void add(int number) {
            store.add(number);
            numMap.put(number, numMap.getOrDefault(number, 0) + 1);
        }

        /**
         * Find if there exists any pair of numbers which sum is equal to the value.
         */
        public boolean find(int value) {

            for (int num : store) {
                if (value - num == num) {
                    if (numMap.getOrDefault(num, 0) > 1) {
                        return true;
                    }
                } else {
                    if (numMap.containsKey(value - num)) {
                        return true;
                    }
                }
            }
            return false;
        }
    }


    class DLL {
        int val;
        int key;
        DLL prev;
        DLL next;

        DLL(int key, int x) {
            this.key = key;
            this.val = x;

        }
    }

    class LRUCache {
        private int capacity;
        private Map<Integer, DLL> keyMap = null;
        private DLL head;
        private DLL tail;
        int count = 0;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.keyMap = new HashMap<>();
        }

        public int get(int key) {
            DLL value = keyMap.get(key);
            if (value == null) {
                return -1;
            }
            if (value.prev == null) {
                return value.key;
            } else if (value.next == null) {
                value.prev.next = null;
            } else {
                value.prev.next = value.next;
            }
            value.next = head;
            head.prev = value;
            if (head.next == null) {
                tail = head;
            }
            head = value;
            return value.val;
        }

        public void put(int key, int value) {
            DLL node = new DLL(key, value);
            if (head == null) {
                head = node;
            } else {
                node.next = head;
                head.prev = node;
                head = node;
            }
            if (head.next == null) {
                tail = head;
            }

            keyMap.put(key, node);
            count++;
            if (count > capacity) {
                keyMap.remove(tail.key);
                tail = tail.prev.next;
                tail.next = null;
                count--;
            }
        }
    }

    class NumArray {

        private int sums[];

        public NumArray(int[] nums) {
            sums = new int[nums.length];
            int sum = 0;
            for (int i = 0; i < nums.length; i++) {
                sum += nums[i];
                sums[i] = sum;
            }
        }

        public int sumRange(int i, int j) {
            if (i == 0) {
                return sums[j];
            }
            return sums[j] - sums[i - 1];
        }
    }


    static void decToBinary(long n) {
        // array to store binary number
        long[] binaryNum = new long[1000];

        // counter for binary array
        int i = 0;
        while (n > 0) {
            // storing remainder in binary array
            binaryNum[i] = n % 2;
            n = n / 2;
            i++;
        }

        // printing binary array in reverse order
        for (int j = i - 1; j >= 0; j--)
            System.out.print(binaryNum[j]);

        System.out.println();
    }

    /**
     * 101
     * 1
     * 001
     *
     * @param n
     * @return
     */
    public int reverseBits(long n) {
        int start = 0;
        int end = 63;
        long res = 0;
        long intMax = 2147483648l;
//        while((n & (intMax >>> end)) == 0){
//            end++;
//        }
//        int tempEnd = end;
        while (start <= 63 / 2) {
            long fromStart = n & (1 << start);
            long fromEnd = (n & (intMax >> start));
            res = res | (fromStart << (31 - start * 2) | fromEnd >> (31 - start * 2));
            decToBinary(res);
            start++;
        }
        return (int) res;
    }

    public int hammingWeight(long n) {
        int hw = 0;
        int len = 0;
        int andVal = 1;
        while (len <= 31) {

            if ((n & andVal) != 0) {
                hw++;
            }
            len++;
            andVal = andVal << 1;
        }
        return hw;
    }

    /**
     * Input: [1,2,3,null,5,null,4]
     * Output: [1, 3, 4]
     * Explanation:
     * <p>
     * 1            <---
     * /   \
     * 2     3         <---
     * \     \
     * 5     4       <---
     *
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {

        Map<Integer, Integer> res = new HashMap<>();
        List<Integer> resList = new ArrayList<>();
        rightSideView(root, 0, res, resList);


        return resList;
    }


    private void rightSideView(TreeNode root, int level, Map<Integer, Integer> res, List<Integer> lis) {
        if (root == null) {
            return;
        }
        if (!res.containsKey(level)) {
            res.put(level, root.val);
            lis.add(root.val);
        }
        if (root.right != null) {
            rightSideView(root.right, level + 1, res, lis);
        }
        if (root.left != null) {
            rightSideView(root.left, level + 1, res, lis);
        }
    }

    /**
     * 3 - 11,4 - 100 , 5 - 101, 6- 110, 7- 111
     *
     * @param
     */


    public int rangeBitwiseAnd(int m, int n) {
        long m1 = (long) m;
        long n1 = (long) n;
        long res = m1;
        for (long i = m1 + 1; i <= n1; i++) {
            if (res == 0) {
                return (int) res;
            }
            res &= i;
        }
        return (int) res;
    }

    /**
     * 1 -> 2 -> 3
     * <p>
     * 1
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(-1);
        ListNode prev = dummy;
        ListNode temp = head;
        while (temp != null) {
            if (temp.val == val) {
                prev.next = temp.next;
            } else {
                prev.next = temp;
                prev = prev.next;
            }

            temp = temp.next;
        }
        return dummy.next;
    }

    /**
     * Input: s = 7, nums = [2,3,1,2,4,3]
     * Output: 2
     * Explanation: the subarray [4,3] has the minimal length under the problem constraint.
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {

        int start = 0;
        int end = 0;
        int sum = 0;
        int minLength = Integer.MAX_VALUE;
        while (end <= nums.length) {
            if (sum < s) {
                if (end < nums.length) {
                    sum += nums[end];
                }
                end++;
            } else {
                if (end - start < minLength) {
                    minLength = end - start;
                }
                sum -= nums[start];
                start++;
            }
        }
        return minLength == Integer.MAX_VALUE ? 0 : minLength;
    }


    static class Trie {

        private boolean isCompleteword;
        private Map<Character, Trie> trie;

        Trie() {
            trie = new HashMap<>();
            isCompleteword = false;
        }

        public Trie addCharacter(char ch) {
            if (trie.containsKey(ch)) {
                return trie.get(ch);
            }
            Trie newTrie = new Trie();
            trie.put(ch, newTrie);
            return newTrie;
        }

        public Map<Character, Trie> getTrie() {
            return trie;
        }

        public Trie isCharacterPresent(char ch) {
            return trie.get(ch);
        }

        public boolean isCompleteWord() {
            return trie.isEmpty();
        }

        public boolean isCompleteword() {
            return isCompleteword;
        }

        public void setCompleteword(boolean completeword) {
            isCompleteword = completeword;
        }


        public int length() {
            return trie.size();
        }
    }


    class WordDictionary {

        private Trie forward;
        private Trie backward;

        public WordDictionary() {
            forward = new Trie();
            backward = new Trie();
        }

        /**
         * Adds a word into the data structure.
         */
        public void addWord(String word) {
            if (word == null || word.trim().equals("")) {
                return;
            }
            Trie temp = forward;
            for (char ch : word.toCharArray()) {
                temp = temp.addCharacter(ch);
            }
            temp.setCompleteword(true);


        }


        /**
         * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
         */
        public boolean search(String word) {
            return search(word, 0, forward);
        }


        public boolean search(String word, int index, Trie trie) {
            if (index >= word.length() && trie.isCompleteword()) {
                return true;
            }
            if (index >= word.length()) {
                return false;
            }
            if (word.charAt(index) == '.') {
                for (Trie uTrie : trie.getTrie().values()) {
                    if (search(word, index + 1, uTrie)) {
                        return true;
                    }
                }
                return false;
            } else {
                Trie newTrie = trie.getTrie().get(word.charAt(index));
                if (newTrie == null) {
                    return false;
                } else {
                    return search(word, index + 1, newTrie);
                }
            }

        }
    }

    /**
     * Input: [2,3,2]
     * Output: 3
     * Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2),
     * because they are adjacent houses.
     *
     * @param nums
     * @return
     */
    public int rob_(int[] nums) {

        return rob1(nums, 0);
    }


    public int rob1(int[] nums, int index) {
        if (index % nums.length < index) {
            return 0;
        }
        return Math.max(rob(nums, index + 2) + nums[index], rob(nums, index + 1));
    }

    /**
     * Input: k = 3, n = 7
     * Output: [[1,2,4]]
     * <p>
     * <p>
     * Input: k = 3, n = 9
     * Output: [[1,2,6], [1,3,5], [2,3,4]]
     * <p>
     * 9
     * 1,2,3,4,5,6,7,8,9
     *
     * @param k
     * @param n
     * @return
     */

    private List<List<Integer>> integerList;

    public List<List<Integer>> combinationSum3(int k, int n) {
        integerList = new ArrayList<List<Integer>>();

        combinationSum3(k, 1, 0, n, new ArrayList());

        return integerList;
    }


    private void combinationSum3(int k, int index, int len, int n, List<Integer> lis) {
        if (len == k && n == 0) {
            integerList.add(new ArrayList(lis));
        } else {
            if (index > 9 || n - index < 0) {
                return;
            }
            for (int i = index; i < 10; i++) {
                Integer inte = new Integer(i);
                lis.add(inte);
                combinationSum3(k, i + 1, len + 1, n - i, lis);
                lis.remove(inte);
            }
        }

    }


    public boolean containsDuplicate(int[] nums) {
        Set<Integer> intSet = new HashSet<>();
        for (int num : nums) {
            if (intSet.contains(num)) {
                return true;
            }
            intSet.add(num);
        }
        return false;
    }


    class Tuple_1 implements Comparable<Tuple_1> {
        int index;
        int data;

        Tuple_1(int index, int data) {
            this.index = index;
            this.data = data;
        }

        @Override
        public int compareTo(Tuple_1 o) {
            return this.data - o.data;
        }
    }


    /**
     * Input: nums = [1,2,3,1], k = 3, t = 0
     * Output: true
     * <p>
     * <p>
     * ()()()()()()()
     *
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
//       List<Tuple_1> set = new ArrayList<>();
//        for(int i=0;i<nums.length;i++){
//           set.add(new Tuple_1(i,nums[i]));
//       }
//
//       Collections.sort(set);
//
//        int start = 0;
//        int end = nums.length-1;
//        while (start < end){
//            if(Math.abs(set.get(start).data - set.get(end).data) <= t && Math.abs(end-start) <=k ){
//                return true;
//            }
//            else if(Math.abs(set.get(start).data - set.get(end).data) >  )
//
//
//        }


        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (Math.abs((long) nums[i] - (long) nums[j]) <= t && (long) Math.abs(i - j) <= k) {
                    return true;
                }
            }
        }
        return false;

    }

    /**

     * <p>
     * A+B*C
     * C*B+A
     * <p>
     * CB*A+
     * +A*BC
     */

    public int calculate(String s) {
        if (s == null) {
            return 0;
        }

        StringBuilder sb = new StringBuilder(s.trim());
        s = sb.reverse().toString();

        Map<Character,Integer> signList = new HashMap<>();
        signList.put('+',1);
        signList.put('-',0);
        signList.put('*',3);
        signList.put('/',4);

        int mul = 1;
        int num = 0;

        String postFix = "";
        Stack<Character> stk = new Stack<>();
        for(char ch:s.toCharArray()){
            if(ch == ' '){
                continue;
            }
            if(signList.containsKey(ch)){
                while (!stk.isEmpty() && signList.get(stk.peek()) > signList.get(ch)){
                    postFix = postFix + stk.pop();
                }
                stk.push(ch);

            }else{
                postFix = postFix + ch;

            }
        }
        while (!stk.isEmpty()){
            postFix = postFix + stk.pop();
        }
Stack<Integer> res = new Stack<>();
       for(char ch:postFix.toCharArray()){

           if(!signList.containsKey(ch)){
               res.push(ch - '0');
           }else{

               switch (ch){
                   case '+':
                       res.push((res.pop()) +(res.pop()));
                       break;
                   case '-':
                       res.push(res.pop() - res.pop());
                       break;
                   case '*':
                       res.push(res.pop()*res.pop());
                       break;
                   case '/':
                       res.push(res.pop()/res.pop());
                       break;

               }
           }

       }
       int fres = 0;

       while (!res.isEmpty()){
           fres = fres*10 + res.pop();
       }
       return fres;
    }





    class MyQueue {
            private Stack<Integer> stack1 = null;
            private Stack<Integer> stack2 = null;
        /** Initialize your data structure here. */
        public MyQueue() {
            stack1 = new Stack<>();
            stack2 = new Stack<>();
        }

        /** Push element x to the back of queue. */
        public void push(int x) {
           stack1.push(x);
        }

        /** Removes the element from in front of queue and returns that element. */
        public int pop() {
            if(!stack2.isEmpty()){
                return stack2.pop();
            }
            else{
                while (!stack1.isEmpty()){
                    stack2.push(stack1.pop());
                }
            }
            return stack2.pop();
        }

        /** Get the front element. */
        public int peek() {
            if(!stack2.isEmpty()){
                return stack2.peek();
            }
            else{
                while (!stack1.isEmpty()){
                    stack2.push(stack1.pop());
                }
            }
            return stack2.peek();

        }

        /** Returns whether the queue is empty. */
        public boolean empty() {
            return stack1.isEmpty() && stack2.isEmpty();
        }
    }

    public void deleteNode(ListNode node) {
        if(node == null){
            return;
        }
            ListNode temp = node;
            ListNode prev = null;
            while (temp.next!=null){
                temp.val = temp.next.val;
                prev = temp;
                temp = temp.next;
            }
            prev.next = null;
    }

    public int[] productExceptSelf(int[] nums) {
         int product = 1;
         for(int num : nums){
             product *= num;
         }

         for(int i=0;i<nums.length;i++){
             nums[i] = product/nums[i];
         }
         return nums;
    }

    /**
     *
     * [
     *   [1,   4,  7, 11, 15],
     *   [2,   5,  8, 12, 19],
     *   [3,   6,  9, 16, 22],
     *   [10, 13, 14, 17, 24],
     *   [18, 21, 23, 26, 30]
     * ]
     *
     *
     * @param matrix
     * @param target
     * @return
     */

    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix.length == 0){
            return false;
        }
        int row =0;
        int col = matrix[0].length-1;
        while(row >=0 && row < matrix.length && col >=0 && col < matrix[row].length)
        {
            if(matrix[row][col] == target){
                return true;
            }else if(matrix[row][col] > target){
                col--;
            }else if(matrix[row][col] < target){
                row++;
            }
        }
    return false;
    }

    /**
     *
     *  2 - 1 - 1
     *
     * Input: "((2-1)-1)"
     * Output: [0, 2]
     * Explanation:
     * ((2-1)-1) = 0
     * (2-(1-1)) = 2
     *
     *
     * @param input
     * @return
     */
    public List<Integer> diffWaysToCompute(String input) {
        List<Integer> arrList = new ArrayList<>();
        if(!(input.contains("*") || input.contains("+") || input.contains("-") || input.contains("/"))){
            arrList.add(Integer.parseInt(input));
            return arrList;
        }
        for(int i=0;i<input.length();i++){
            char ch = input.charAt(i);
            if(ch == '-' || ch == '*' || ch == '/' || ch == '+'){
                List<Integer> left = diffWaysToCompute(input.substring(0,i));
                List<Integer> right = diffWaysToCompute(input.substring(i+1));
                for(Integer p1 : left){
                    for(Integer p2: right){
                        switch (ch){
                            case '-':
                                arrList.add(p1-p2);
                                break;
                            case '+':
                                arrList.add(p1+p2);
                                break;
                            case '*':
                                arrList.add(p1*p2);
                                break;
                            case '/':
                                arrList.add(p1/p2);
                                break;
                        }
                    }
                }
            }
        }
        return arrList;
    }



    //   {"a","a","b","b"}
    public int shortestDistance(String[] words, String word1, String word2) {
           int i1 =-1;int i2 = -1;
           int minDis = Integer.MAX_VALUE;
           for(int i=0;i<words.length;i++){
               if(words[i].equalsIgnoreCase(word1)){
                   i1 = i;
               }else if(words[i].equalsIgnoreCase(word2)){
                   i2 = i;
               }
               if(i1!=-1&&i2!=-1) {
                   minDis = Math.min(minDis, Math.abs(i1 - i2));
               }
           }
           return minDis;
    }




    static class WordDistance {
        private Map<String,List<Integer>> stringListMap = null;

        public WordDistance(String[] words) {
            stringListMap = new HashMap<>();
            for(int i=0;i<words.length;i++){
                List<Integer> posList = stringListMap.getOrDefault(words[i], new ArrayList<>());
                posList.add(i);
                stringListMap.put(words[i],posList);
            }
        }

        public int shortest(String word1, String word2) {
            int minDis = Integer.MAX_VALUE;
            List<Integer> loc1 = stringListMap.get(word1);
            List<Integer> loc2 = stringListMap.get(word2);
            int index1 = 0;
            int index2 = 0;
            while (index1 < loc1.size() && index2 < loc2.size()){
                minDis = Math.min(minDis,Math.abs(loc1.get(index1) - loc2.get(index2)));
                if(loc1.get(index1) < loc2.get(index2)){
                    index1++;
                }else{
                    index2++;
                }
            }
            return minDis;
        }
    }





    public int shortestWordDistance1(String[] words, String word1, String word2) {

        int i1 =-1;int i2 = -1;
        int minDis = Integer.MAX_VALUE;
        for(int i=0;i<words.length;i++){
            if(word1.equalsIgnoreCase(word2) && words[i].equalsIgnoreCase(word1)){
               if(i1 == -1){
                   i1 = i;
               }else if(i2 == -1){
                   i2 = i;
                   minDis = Math.min(minDis,Math.abs(i1-i2));
                   i1 = i2;
                   i2 = -1;
               }
            }
           else if(words[i].equalsIgnoreCase(word1)){
                i1 = i;
            }else if(words[i].equalsIgnoreCase(word2)){
                i2 = i;
            }
            if(i1!=-1&&i2!=-1) {
                minDis = Math.min(minDis, Math.abs(i1 - i2));
            }
        }
        return minDis;
    }





    public List<String> helper( int n,int m){
        List<String> stringList = new ArrayList<>();
        if(n <= 0){
            stringList.add("");
            return stringList;
        }else if(n == 1){
            stringList.addAll(Arrays.asList("0","1","8"));
            return stringList;
        }else {
            List<String> res = helper(n-2,m);
            for(String s:res){
                if(m!=n){
                    stringList.add("0"+s+"0");
                }
                stringList.add("1"+s+"1");
                stringList.add("6"+s+"9");
                stringList.add("8"+s+"8");
                stringList.add("9"+s+"6");
            }
            return stringList;
        }

    }

    /**
     *
     * Given an array of numbers, verify whether it is the correct preorder traversal sequence of a binary search tree.
     *
     * You may assume each number in the sequence is unique.
     *
     * Consider the following binary search tree:
     *
     *      5
     *     / \
     *    2   6
     *   / \
     *  1   3
     * Example 1:
     *
     * Input: [5,2,6,1,3]
     * Output: false
     * Example 2:
     *5,3
     * Input: [5,2,1,3,6]
     * Output: true
     *
     *
     * @param preorder
     * @return
     */
    public boolean verifyPreorder1(int[] preorder) {
        Stack<Integer> stack = new Stack<>();
        int curr = Integer.MIN_VALUE;
        for(int num:preorder){
            if(num < curr){
                return false;
            }
            while (!stack.isEmpty() && stack.peek() < num){
                curr = Math.max(stack.pop(), curr);
            }
                stack.push(num);
        }
       return true;
    }


    public boolean verifyPreorder(int[] preorder) {
        int curr = Integer.MIN_VALUE;
        int index = -1;

        for(int i=0;i < preorder.length; i++){
            if(preorder[i] < curr){
                return false;
            }
            while(index>=0 && preorder[index] < preorder[i]) {
                curr = Math.max(preorder[index],curr);
                index--;
            }
            index++;
        }
      return true;
    }

    static class ValidWordAbbr {

        Set<String> abbrvs;
        Set<String> words;
        public ValidWordAbbr(String[] dictionary) {
            abbrvs = new HashSet<>();
            words = new HashSet<>();
            for(String word:dictionary){
                abbrvs.add(abbr(word));
                words.add(word);
            }
        }

        private String abbr(String word){
            int len = word.length() - 2;
            if(len <= 0){
                return word;
            }
            return ""+word.charAt(0)+len+word.charAt(word.length()-1);
        }

        public boolean isUnique(String word) {
            return !abbrvs.contains(abbr(word));
        }
    }

    public static void main(String[] args) {

   int[] preOrder = {5,2,6,1,3};
        Solution sol = new Solution();
        ValidWordAbbr validWordAbbr = new ValidWordAbbr(new String[]{"hello"});
        System.out.println(validWordAbbr.isUnique("hello"));

    }
}




