# Reference

Look-up material. Nothing here teaches — the [guides](../guides/index.md) do that.

<div class="pl-cards pl-cards--tight" markdown>

<div class="pl-card" markdown>
### [Configuration](configuration.md)
The configuration keys Platoon owns, what each one does, and which backend it applies to.
</div>

<div class="pl-card" markdown>
### [CLI](cli.md)
Training, inference and rollout-inspection entrypoints, with the flags each one takes.
</div>

<div class="pl-card" markdown>
### [FAQ](faq.md)
Short answers to the questions that come up first.
</div>

</div>

Two neighbours worth knowing about: [component contracts](../architecture/components.md) gives the
signature each registry kind expects, and the [plugin catalog](../plugins/catalog.md) lists the task
and capability plugins that ship with the repository.

!!! note

    Where a key here disagrees with the dataclasses in the source tree, the source wins.
