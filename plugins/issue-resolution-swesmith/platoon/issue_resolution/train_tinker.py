"""Issue resolution training script with Tinker backend."""
import asyncio
import logging
import sys
from pathlib import Path

def configure_plain_logging() -> None:
    """Install plain stderr logging before dependencies auto-configure handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    logging.getLogger("platoon").setLevel(logging.INFO)
    logging.getLogger("openhands").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


configure_plain_logging()

from datasets import Dataset
from platoon.issue_resolution.rollout import run_rollout
from platoon.issue_resolution.tasks import get_task, load_data
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.fastapi_litellm_proxy import FastAPILiteLLMTinkerHTTPProxyServer
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

# logging.basicConfig(
#     level=logging.WARNING,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# logging.getLogger("platoon").setLevel(logging.DEBUG)
# logging.getLogger("httpx").setLevel(logging.WARNING)

async def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "train_issue_resolution_fft_tinker.yaml"
    config, raw_config = load_config(
        args=args,
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=str(default_config),
    )
    instance_ids = [
"getmoto__moto.694ce1f4.pr_8478",
"pdfminer__pdfminer.six.1a8bd2f7.func_basic__7m2sf1xf",
"facebookresearch__hydra.0f03eb60.func_pm_ctrl_shuffle__bt2xmk69",
"oauthlib__oauthlib.1fd52536.func_basic__ql8z45kn",
"kayak__pypika.1c9646f0.func_basic__52qyvjch",
"spulec__freezegun.5f171db0.func_basic__hyt08iv0",
"tkrajina__gpxpy.09fc46b3.func_basic__wofex2b0",
"oauthlib__oauthlib.1fd52536.func_basic__nong5iwt",
"pandas-dev__pandas.95280573.func_pm_class_rm_base__gqb4hx1c",
"dask__dask.5f61e423.lm_rewrite__kex0qbbb",
"python-openxml__python-docx.0cf6d71f.func_basic__efne04gd",
"msiemens__tinydb.10644a0e.func_pm_ctrl_shuffle__ntejgr8m",
"arrow-py__arrow.1d70d009.func_pm_class_rm_base__5hdqgjdc",
"pallets__markupsafe.620c06c9.func_basic__sala7a01",
"pyutils__line_profiler.a646bf0f.func_pm_remove_assign__7ldqjz8e",
"pydantic__pydantic.acb0f10f.func_pm_remove_assign__4p0bl2q9",
"scanny__python-pptx.278b47b1.func_basic__n1mrychf",
"pyutils__line_profiler.a646bf0f.func_basic__vx51jkod",
"encode__starlette.db5063c2.lm_rewrite__ciiqqrf3",
"jsvine__pdfplumber.02ff4313.func_pm_ctrl_invert_if__o8ppw4ot",
"msiemens__tinydb.10644a0e.combine_file__3nxtrr1s",
"tkrajina__gpxpy.09fc46b3.func_pm_op_break_chains__v2tkfcqn",
"pdfminer__pdfminer.six.1a8bd2f7.func_basic__leu68vmh",
"Suor__funcy.207a7810.func_basic__ep4gx2nd",
"tkrajina__gpxpy.09fc46b3.func_pm_op_swap__f2gu0488",
"pdfminer__pdfminer.six.1a8bd2f7.func_basic__gqqn47rp",
"oauthlib__oauthlib.1fd52536.func_basic__jfzqihyy",
"pdfminer__pdfminer.six.1a8bd2f7.func_pm_ctrl_shuffle__88eyw6mh",
"cantools__cantools.0c6a7871.func_basic__veayvqcz",
"pandas-dev__pandas.95280573.func_pm_ctrl_invert_if__d11wzbl9",
"jsvine__pdfplumber.02ff4313.func_pm_ctrl_shuffle__i8rn7hxd",
"scanny__python-pptx.278b47b1.func_pm_ctrl_shuffle__0bn1j8sh",
"python-openxml__python-docx.0cf6d71f.func_basic__kf8egr68",
"arrow-py__arrow.1d70d009.func_basic__fw5h8jjk",
"spulec__freezegun.5f171db0.pr_533",
"pallets__click.fde47b4b.func_basic__b6d99806",
"pandas-dev__pandas.95280573.func_pm_remove_assign__eof6pluz",
"msiemens__tinydb.10644a0e.func_basic__2e18lkf6",
"kayak__pypika.1c9646f0.func_basic__cv5seh3a",
"modin-project__modin.8c7799fd.func_pm_remove_assign__30iq7hjw",
"Suor__funcy.207a7810.lm_rewrite__xpj94mrn",
"tornadoweb__tornado.d5ac65c1.func_pm_class_rm_base__fnbgli8e",
"pdfminer__pdfminer.six.1a8bd2f7.func_pm_ctrl_shuffle__p83c0pkl",
"kayak__pypika.1c9646f0.func_pm_ctrl_invert_if__v36rwvkt",
"kayak__pypika.1c9646f0.func_basic__4jvtbhp9",
"pandas-dev__pandas.95280573.func_pm_ctrl_invert_if__yluwn5np",
"pandas-dev__pandas.95280573.func_pm_remove_assign__kz178ntn",
"cantools__cantools.0c6a7871.func_basic__jac9y2h5",
"msiemens__tinydb.10644a0e.func_basic__xewv2osl",
"pydantic__pydantic.acb0f10f.pr_11364",
"pallets__markupsafe.620c06c9.func_basic__fpb7w0tc",
"python-openxml__python-docx.0cf6d71f.func_basic__beqal72v",
"jsvine__pdfplumber.02ff4313.func_pm_ctrl_shuffle__jqw6e553",
"encode__starlette.db5063c2.func_basic__cmdcyxyi",
"Suor__funcy.207a7810.func_basic__2htedgfv",
"python-openxml__python-docx.0cf6d71f.func_basic__pnjrpt6p",
"encode__starlette.db5063c2.func_basic__841oanl3",
"msiemens__tinydb.10644a0e.func_basic__moedefto",
"spulec__freezegun.5f171db0.pr_539",
"pdfminer__pdfminer.six.1a8bd2f7.lm_rewrite__lv1ja7kx",
"facebookresearch__hydra.0f03eb60.func_pm_ctrl_shuffle__smruwdau",
"scrapy__scrapy.35212ec5.func_pm_ctrl_shuffle__vvydgv69",
"pyutils__line_profiler.a646bf0f.func_basic__1cp2dq7f",
"iterative__dvc.1d6ea681.lm_rewrite__769syjv9",
"pallets__click.fde47b4b.func_basic__95733822",
"Suor__funcy.207a7810.func_basic__8vv3z712",
"getmoto__moto.694ce1f4.func_pm_class_rm_funcs__srg0pb98",
"python-openxml__python-docx.0cf6d71f.func_basic__nkwrl6zk",
"python-openxml__python-docx.0cf6d71f.func_basic__1hyp9o5a",
"msiemens__tinydb.10644a0e.func_pm_ctrl_invert_if__935wmg72",
"msiemens__tinydb.10644a0e.func_basic__rqzp6jrz",
"pallets__click.fde47b4b.func_pm_ctrl_invert_if__0qqby88o",
"pandas-dev__pandas.95280573.func_pm_ctrl_invert_if__gdcuj5te",
"spulec__freezegun.5f171db0.func_basic__ny5bncew",
"pallets__markupsafe.620c06c9.func_basic__x3obno8n",
"kayak__pypika.1c9646f0.func_basic__9l5f6k6u",
"kayak__pypika.1c9646f0.func_basic__dgrtcmwz",
"conan-io__conan.86f29e13.func_pm_remove_loop__y59worpf",
"scrapy__scrapy.35212ec5.func_pm_class_rm_base__7odd8hev",
"jsvine__pdfplumber.02ff4313.func_pm_ctrl_shuffle__6f7kopcy",
"django-money__django-money.835c1ab8.lm_rewrite__ddzxlx5n",
"dask__dask.5f61e423.pr_10521",
"encode__starlette.db5063c2.func_basic__35hnxfnd",
"pyutils__line_profiler.a646bf0f.lm_rewrite__pvsbhrkn",
"scanny__python-pptx.278b47b1.combine_file__ow8zpk04",
"dask__dask.5f61e423.pr_7656",
"python-openxml__python-docx.0cf6d71f.func_basic__tkamt53f",
"Suor__funcy.207a7810.func_basic__gqw31m2m",
"jsvine__pdfplumber.02ff4313.func_pm_ctrl_invert_if__7uwa6dkz",
"encode__starlette.db5063c2.pr_2583",
"Suor__funcy.207a7810.lm_rewrite__k68y697t",
"oauthlib__oauthlib.1fd52536.func_basic__q4ugibgr",
"django-money__django-money.835c1ab8.func_basic__080po5jt",
"cantools__cantools.0c6a7871.func_basic__o62d6coq",
"getmoto__moto.694ce1f4.func_pm_class_rm_funcs__pveltwwm",
"tkrajina__gpxpy.09fc46b3.func_pm_op_change_const__o5nz6xj0",
"dask__dask.5f61e423.lm_rewrite__mkzy35wr",
"pallets__markupsafe.620c06c9.pr_402",
"spulec__freezegun.5f171db0.func_pm_ctrl_shuffle__tmezuzc1",
"msiemens__tinydb.10644a0e.func_basic__eyiycmk3",
"spulec__freezegun.5f171db0.func_pm_ctrl_shuffle__e86ei5fl",
"pyutils__line_profiler.a646bf0f.lm_rewrite__0oz7n2aa",
"arrow-py__arrow.1d70d009.func_pm_ctrl_shuffle__bsh5nbw2",
"arrow-py__arrow.1d70d009.lm_rewrite__76bcuvzu",
"msiemens__tinydb.10644a0e.func_basic__thkyx351",
"getmoto__moto.694ce1f4.func_pm_op_change_const__alol1trj",
"tornadoweb__tornado.d5ac65c1.func_pm_ctrl_invert_if__q1xpqtwd",
"spulec__freezegun.5f171db0.func_pm_ctrl_shuffle__ql5q3yhc",
"msiemens__tinydb.10644a0e.func_pm_ctrl_shuffle__8a4amosv",
"tkrajina__gpxpy.09fc46b3.func_pm_ctrl_shuffle__wad66d1k",
"tkrajina__gpxpy.09fc46b3.func_pm_remove_assign__4ysq0yw6",
"pandas-dev__pandas.95280573.func_pm_remove_wrapper__3bgagt0d",
"encode__starlette.db5063c2.func_basic__55w98645",
"arrow-py__arrow.1d70d009.lm_rewrite__vxpm6hj5",
"arrow-py__arrow.1d70d009.func_pm_ctrl_shuffle__lr8th6r2",
"Suor__funcy.207a7810.func_basic__ku01y3x1",
"pdfminer__pdfminer.six.1a8bd2f7.func_basic__v5s8zjd6",
"getmoto__moto.694ce1f4.func_pm_op_swap__mdwc1boa",
"Suor__funcy.207a7810.lm_rewrite__01e30g2p",
"pydantic__pydantic.acb0f10f.func_pm_ctrl_shuffle__5nmtjrxp",
"Suor__funcy.207a7810.func_basic__1dijeo7t",
"modin-project__modin.8c7799fd.func_pm_op_change__sehsbp99",
"modin-project__modin.8c7799fd.func_pm_ctrl_shuffle__gg9iieck",
"python-openxml__python-docx.0cf6d71f.func_basic__oydbgg2e",
"kayak__pypika.1c9646f0.combine_file__0255t16v",
"encode__starlette.db5063c2.func_basic__obkkiost",
"modin-project__modin.8c7799fd.func_pm_class_rm_funcs__fb82om6g",
"encode__starlette.db5063c2.func_pm_ctrl_shuffle__59xsp6vh",
"tkrajina__gpxpy.09fc46b3.func_basic__4najw2qo",
"scrapy__scrapy.35212ec5.func_pm_remove_cond__pjogstll",
"msiemens__tinydb.10644a0e.func_basic__fg8psbzc",
"cantools__cantools.0c6a7871.func_pm_ctrl_invert_if__5rxooyuv",
"encode__starlette.db5063c2.func_basic__a3wuo4au",
"msiemens__tinydb.10644a0e.func_pm_ctrl_shuffle__zdljkch7",
"django-money__django-money.835c1ab8.lm_rewrite__37szs5s4",
"scanny__python-pptx.278b47b1.func_basic__cg7faspb",
"encode__starlette.db5063c2.func_pm_ctrl_invert_if__pct7ov1t",
"scanny__python-pptx.278b47b1.func_basic__a1snzbvp",
"scanny__python-pptx.278b47b1.func_basic__xt20srm6",
"scanny__python-pptx.278b47b1.func_basic__53p94gd9",
"arrow-py__arrow.1d70d009.func_pm_ctrl_invert_if__eqpaz75s",
"conan-io__conan.86f29e13.pr_11283",
"oauthlib__oauthlib.1fd52536.lm_rewrite__e551pdwl",
"pyutils__line_profiler.a646bf0f.func_basic__r4uud43s",
"kayak__pypika.1c9646f0.func_basic__1vp3ltxf",
"msiemens__tinydb.10644a0e.func_pm_ctrl_shuffle__6c6nhbz9",
"oauthlib__oauthlib.1fd52536.combine_file__xmiq50ne",
"pydantic__pydantic.acb0f10f.pr_4455",
"cantools__cantools.0c6a7871.combine_file__8cdnpuum",
"cantools__cantools.0c6a7871.func_basic__g6zdkcsn",
"scrapy__scrapy.35212ec5.func_pm_ctrl_invert_if__p3o6p5zo",
"jsvine__pdfplumber.02ff4313.func_pm_ctrl_shuffle__jhjo0van",
"scanny__python-pptx.278b47b1.func_basic__saq13lj8",
"pallets__markupsafe.620c06c9.func_basic__rgp7xs9w",
"msiemens__tinydb.10644a0e.func_pm_ctrl_shuffle__2gk7hfpy",
"tkrajina__gpxpy.09fc46b3.func_pm_op_change__g35fy9ez",
"python-openxml__python-docx.0cf6d71f.func_basic__y59awy2x",
"scanny__python-pptx.278b47b1.func_basic__m0wx67w5",
"arrow-py__arrow.1d70d009.func_pm_op_swap__sdhl3dkf",
"msiemens__tinydb.10644a0e.func_pm_class_rm_funcs__s05hyh19",
"cantools__cantools.0c6a7871.func_basic__jfbj0yuf",
"arrow-py__arrow.1d70d009.func_pm_ctrl_shuffle__zo1id9uk",
"python-openxml__python-docx.0cf6d71f.func_basic__kh2lbmul",
"oauthlib__oauthlib.1fd52536.func_pm_class_rm_funcs__auo4n3db",
"spulec__freezegun.5f171db0.func_pm_ctrl_shuffle__hq96g33n",
"oauthlib__oauthlib.1fd52536.func_pm_ctrl_shuffle__6u465ug7",
"pallets__markupsafe.620c06c9.func_basic__oepksuqv",
"django-money__django-money.835c1ab8.func_basic__rpkyl848",
"arrow-py__arrow.1d70d009.func_basic__w1skinei",
"spulec__freezegun.5f171db0.func_pm_ctrl_invert_if__sl25599x",
"pdfminer__pdfminer.six.1a8bd2f7.func_pm_ctrl_invert_if__rhrmdfgl",
"oauthlib__oauthlib.1fd52536.func_basic__r62x6g2b",
"pallets__markupsafe.620c06c9.func_basic__9eer257s",
"pydantic__pydantic.acb0f10f.pr_7891",
"spulec__freezegun.5f171db0.pr_538",
"tkrajina__gpxpy.09fc46b3.func_pm_ctrl_shuffle__fs9ested",
"getmoto__moto.694ce1f4.func_pm_class_rm_base__mzlrypww",
"encode__starlette.db5063c2.func_basic__40piio9i",
"cantools__cantools.0c6a7871.func_basic__l1okym77",
"msiemens__tinydb.10644a0e.func_basic__au84v388",
"pallets__markupsafe.620c06c9.func_basic__7k8k15ja",
"scrapy__scrapy.35212ec5.pr_6007",
"msiemens__tinydb.10644a0e.func_pm_ctrl_shuffle__twt15ie9",
"oauthlib__oauthlib.1fd52536.func_basic__zrajljoi",
"tornadoweb__tornado.d5ac65c1.func_pm_remove_assign__xgokfggp",
"spulec__freezegun.5f171db0.func_pm_ctrl_shuffle__kwv8tgpx",
"getmoto__moto.694ce1f4.func_pm_remove_assign__5lvzaecv",
"cantools__cantools.0c6a7871.func_basic__n45iyvgh",
"kayak__pypika.1c9646f0.func_basic__2sdoideg",
"Suor__funcy.207a7810.func_basic__6u9411f8",
"spulec__freezegun.5f171db0.func_basic__uypfx1km",
"scanny__python-pptx.278b47b1.func_basic__doioeabf",
"spulec__freezegun.5f171db0.func_basic__7o2ls9ov",
"cantools__cantools.0c6a7871.func_pm_class_rm_base__shirj7f8",
"cantools__cantools.0c6a7871.func_basic__uehabsy0",
"jsvine__pdfplumber.02ff4313.func_pm_ctrl_invert_if__5xvxen84",
"spulec__freezegun.5f171db0.func_basic__ah18mne0",
"pyutils__line_profiler.a646bf0f.func_pm_class_rm_base__zzm5xszk",
"python-openxml__python-docx.0cf6d71f.func_basic__1zenzf3o",
"arrow-py__arrow.1d70d009.func_pm_ctrl_shuffle__gnef3fqr",
"scanny__python-pptx.278b47b1.func_basic__pu0xkyjr",
"oauthlib__oauthlib.1fd52536.func_basic__uqy2evrq",
"msiemens__tinydb.10644a0e.func_basic__bwrlp83e",
"spulec__freezegun.5f171db0.func_basic__hfap00ae",
"python-openxml__python-docx.0cf6d71f.func_pm_ctrl_shuffle__mz6s17vj",
"encode__starlette.db5063c2.func_basic__h7rvrbgi",
"spulec__freezegun.5f171db0.lm_rewrite__b0ixukh7",
"oauthlib__oauthlib.1fd52536.lm_rewrite__mq7qhqvt",
"python-openxml__python-docx.0cf6d71f.func_basic__533ntckf",
"pallets__click.fde47b4b.func_basic__d4dd5544",
"Suor__funcy.207a7810.func_basic__aeks8s4d",
"pandas-dev__pandas.95280573.func_pm_op_break_chains__ihy0d4mv",
"Suor__funcy.207a7810.func_basic__yunfv3yt",
"Suor__funcy.207a7810.func_basic__dto91avw",
"scanny__python-pptx.278b47b1.func_basic__jwx94m16",
"jsvine__pdfplumber.02ff4313.func_pm_op_change_const__d11cey7n",
"pdfminer__pdfminer.six.1a8bd2f7.func_basic__1c5t5wgl",
"arrow-py__arrow.1d70d009.func_basic__isao90u1",
"scanny__python-pptx.278b47b1.func_pm_ctrl_shuffle__v65v968n",
"getmoto__moto.694ce1f4.pr_6536",
"scanny__python-pptx.278b47b1.func_basic__1xp83dqz",
"encode__starlette.db5063c2.func_pm_class_rm_base__5mwvr3jj",
"oauthlib__oauthlib.1fd52536.func_basic__vbflyd3r",
"kayak__pypika.1c9646f0.combine_file__mq1vodv3",
"modin-project__modin.8c7799fd.func_pm_op_swap__a9579o7b",
"Suor__funcy.207a7810.func_basic__49c0rccl",
"conan-io__conan.86f29e13.pr_16429",
"getmoto__moto.694ce1f4.pr_5862",
"modin-project__modin.8c7799fd.pr_7225",
"dask__dask.5f61e423.lm_rewrite__6mq2udnj",
"encode__starlette.db5063c2.pr_2409",
"python-openxml__python-docx.0cf6d71f.func_pm_ctrl_shuffle__rwoc67fy",
"tkrajina__gpxpy.09fc46b3.func_pm_ctrl_shuffle__gqo7oq4k",
"encode__starlette.db5063c2.func_basic__6yem8u40",
"kayak__pypika.1c9646f0.func_pm_ctrl_shuffle__5hk61hr9",
"tkrajina__gpxpy.09fc46b3.func_pm_ctrl_shuffle__kg0bag2z",
"scanny__python-pptx.278b47b1.func_basic__61z3ujel",
"facebookresearch__hydra.0f03eb60.pr_3022",
"django-money__django-money.835c1ab8.combine_file__i1ay87eb",
"oauthlib__oauthlib.1fd52536.func_basic__r1ldhn4f",
"scanny__python-pptx.278b47b1.func_pm_ctrl_shuffle__xduv3tez",
"arrow-py__arrow.1d70d009.func_pm_class_rm_base__m81hj630",
"msiemens__tinydb.10644a0e.func_basic__g0u14ntl",
"pdfminer__pdfminer.six.1a8bd2f7.func_basic__clagqaqv",
"pallets__click.fde47b4b.func_basic__bea5e562",
"encode__starlette.db5063c2.func_basic__ti1d811h",
"facebookresearch__hydra.0f03eb60.pr_2361",
"Suor__funcy.207a7810.func_basic__dd6c7obd",
"pdfminer__pdfminer.six.1a8bd2f7.func_pm_class_rm_base__7s30p09b",
"pallets__markupsafe.620c06c9.pr_379",
"arrow-py__arrow.1d70d009.func_pm_ctrl_shuffle__vaslaiy9",
"cantools__cantools.0c6a7871.func_pm_ctrl_shuffle__9saaz81b",
"dask__dask.5f61e423.pr_10789",
"tkrajina__gpxpy.09fc46b3.func_basic__gwaqjhgv",
"getmoto__moto.694ce1f4.pr_5699",
"spulec__freezegun.5f171db0.func_pm_remove_wrapper__s7fminim",
"kayak__pypika.1c9646f0.combine_file__7tqllwvr"
]
    train_datamap, val_datamap = load_data()
    train_dataset = Dataset.from_list([{ "task_id": x } for x in train_datamap.keys() if train_datamap[x].id in instance_ids])
    # val_dataset = Dataset.from_list([{ "task_id": x } for x in val_datamap.keys()])
    # Create trainer and run with context manager for proper cleanup
    print(f"Training dataset size: {len(train_dataset)}")
    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    async with trainer:
        old_model_name = trainer.model_info.model_name
        old_base_url = trainer.model_info.base_url
        old_api_key = trainer.model_info.api_key
        tinker_proxy = FastAPILiteLLMTinkerHTTPProxyServer(
            litellm_model_name=old_model_name,
            context_window_length=trainer.model_info.llm.context_window_length,
        )
        tinker_proxy.start()
        trainer.model_info.model_name = tinker_proxy.model_name
        trainer.model_info.base_url = tinker_proxy.base_url
        trainer.model_info.api_key = tinker_proxy.api_key
        try:
            # Create workflows - use trainer.run_log_path for run-specific output
            train_workflow = GroupRolloutWorkflow(
                rollout_fn=run_rollout,
                get_task_fn=get_task,
                config=config.train.workflow_config,
                model_info=trainer.model_info,
                log_path=trainer.run_log_path,
                stats_scope="train",
                filter_errors=False,
            )
            # Run training
            await trainer.train(
                train_workflow=train_workflow,
                eval_workflow=None,
            )
        finally:
            trainer.model_info.model_name = old_model_name
            trainer.model_info.base_url = old_base_url
            trainer.model_info.api_key = old_api_key
            tinker_proxy.stop()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))