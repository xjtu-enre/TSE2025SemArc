import java.util.*;

public class AWS {
    public static void main(String[] args){
        Integer[] test1 = {3,2,1,23,4,12};
        Integer[] test2 = {2,1,1,12,2,2};
        Integer[] test3 = {6,5,3,1};
        System.out.println("getMinSwaps(test1)"+getMinSwaps(Arrays.asList(test1)));
        System.out.println("getMinSwaps(test2)"+getMinSwaps(Arrays.asList(test2)));
        System.out.println("getMinSwaps(test3)"+getMinSwaps(Arrays.asList(test3)));
        System.out.println("getMaxNegatives(test1)"+getMaxNegatives(Arrays.asList(test1)));
        System.out.println("getMaxNegatives(test2)"+getMaxNegatives(Arrays.asList(test2)));
        System.out.println("getMaxNegatives(test3)"+getMaxNegatives(Arrays.asList(test3)));
    }

    /**
     * https://leetcode.com/problems/minimum-adjacent-swaps-to-make-a-valid-array/description/
     * 
     */
    public static int getMinSwaps(List<Integer> blocks){
        int maxIndex = blocks.size() - 1;//index of maxValue from the right
        int minIndex = 0;//index of minValue from the left
        for(int i = 1; i < blocks.size() ; i++){
            if(blocks.get(minIndex)>blocks.get(i)){//this ones going in order to find the min
                minIndex = i;
            }
            if(blocks.get(maxIndex)<blocks.get(blocks.size() -1-i)){//this ones going reverse to find the max
                maxIndex = blocks.size()-1-i;
            }
        }
        int ans = minIndex+(blocks.size() -1-maxIndex);//distance of each from respective sides is the min of swaps
        if(maxIndex < minIndex){//case when maxIndex and  minIndex are adjacent to each other
            ans--;
        }
        return ans;
    }

    /**
     * determine the max amount of negatives that will allow the cumulative sum to still be positive.
     * 
     */
    public static int getMaxNegatives(List<Integer> PnL){
        if(PnL.size()<=1){
            return 0;
        }
        return backTracking(PnL, 1, PnL.get(0),0);
    }

    public static int backTracking(List<Integer> PnL, int index, int cumSum, int totalCount){
        if(cumSum <= 0){
            return 0;
        }
        if(index == PnL.size()){
            return totalCount;
        }

        return Math.max(backTracking(PnL,index+1,cumSum+PnL.get(index),totalCount),
                        backTracking(PnL,index+1,cumSum+(PnL.get(index)*-1),totalCount+1));
    }
}