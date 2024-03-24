import 'package:flutter/material.dart';
import 'package:instagram_clone/models/user_model.dart';
import 'package:instagram_clone/providers/search/search_provider.dart';
import 'package:instagram_clone/providers/search/search_state.dart';
import 'package:instagram_clone/utils/debounce.dart';
import 'package:instagram_clone/widgets/avatar_widget.dart';
import 'package:provider/provider.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final Debounce debounce = Debounce(milliseconds: 500);

  @override
  void initState() {
    super.initState();
    _clearSearchState();
  }

  void _clearSearchState() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<SearchProvider>().clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    List<UserModel> userModelList = context.watch<SearchState>().userModelList;

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            TextField(
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                labelText: '이름을 입력해주세요',
              ),
              onChanged: (value) {
                debounce.run(() async {
                  if (value.trim().isNotEmpty) {
                    await context
                        .read<SearchProvider>()
                        .searchUser(keyword: value);
                  } else {
                    _clearSearchState();
                  }
                });
              },
            ),
            SizedBox(height: 10),
            Expanded(
              child: GestureDetector(
                onTap: () => FocusScope.of(context).unfocus(),
                child: ListView.builder(
                  itemCount: userModelList.length,
                  itemBuilder: (context, index) {
                    UserModel userModel = userModelList[index];
                    return Container(
                      padding: const EdgeInsets.symmetric(
                        vertical: 8,
                        horizontal: 5,
                      ),
                      child: Row(
                        children: [
                          AvatarWidget(userModel: userModel),
                          SizedBox(width: 10),
                          Text(userModel.name),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
