import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/user_model.dart';

class SearchRepository {
  final FirebaseFirestore firebaseFirestore;

  const SearchRepository({
    required this.firebaseFirestore,
  });

  Future<List<UserModel>> searchUser({
    required String keyword,
  }) async {
    try {
      QuerySnapshot<Map<String, dynamic>> querySnapshot =
          await firebaseFirestore
              .collection('users')
              .where('name', isGreaterThanOrEqualTo: keyword)
              .where('name', isLessThanOrEqualTo: keyword + '\uf7ff')
              .get();

      List<UserModel> userList = querySnapshot.docs
          .map((user) => UserModel.fromMap(user.data()))
          .toList();
      return userList;
    } on FirebaseException catch (e) {
      throw CustomException(
        code: e.code,
        message: e.message!,
      );
    } catch (e) {
      throw CustomException(
        code: 'Exception',
        message: e.toString(),
      );
    }
  }
}
