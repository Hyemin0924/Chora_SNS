import 'package:flutter_state_notifier/flutter_state_notifier.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/user_model.dart';
import 'package:instagram_clone/providers/search/search_state.dart';
import 'package:instagram_clone/repositories/search_repository.dart';

class SearchProvider extends StateNotifier<SearchState> with LocatorMixin {
  SearchProvider() : super(SearchState.init());

  void clear() {
    state = state.copyWith(userModelList: []);
  }

  Future<void> searchUser({
    required String keyword,
  }) async {
    state = state.copyWith(searchStatus: SearchStatus.searching);

    try {
      List<UserModel> userModelList =
          await read<SearchRepository>().searchUser(keyword: keyword);
      state = state.copyWith(
        searchStatus: SearchStatus.success,
        userModelList: userModelList,
      );
    } on CustomException catch (_) {
      state = state.copyWith(searchStatus: SearchStatus.error);
      rethrow;
    }
  }
}
