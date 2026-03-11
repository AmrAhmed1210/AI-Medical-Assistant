namespace MedicalAssistant.Domain.Entities.DoctorsModule;

public class Doctor : BaseEntity
{
    public string Name { get; set; } = string.Empty;

    public string Location { get; set; } = string.Empty;

    public int Experience { get; set; }

    public decimal ConsultationFee { get; set; }

    public double Rating { get; set; }

    public int ReviewCount { get; set; }

    public bool IsAvailable { get; set; }

    public string Bio { get; set; } = string.Empty;

    public string ImageUrl { get; set; } = "default-doctor.png";

    public int SpecialtyId { get; set; }

    public Specialty Specialty { get; set; } = null!;
}